import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0, freeze_base: bool = True):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear base module.")
        self.base = base
        self.r = r
        self.register_buffer(
            'scaling',
            torch.tensor(alpha / r if r > 0 else 0.0),
            persistent=False,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
            self.lora_B = nn.Parameter(torch.empty(base.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, x):
        result = self.base(x)
        if self.r > 0:
            lora = self.dropout(x)
            lora = F.linear(lora, self.lora_A)
            lora = F.linear(lora, self.lora_B) * self.scaling
            result = result + lora
        return result


def _get_parent_module(root: nn.Module, name: str):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
        model: nn.Module,
        targets,
        r: int,
        alpha: int,
        dropout: float,
        only_blocks: bool = True,
):
    targets = tuple(targets)
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if only_blocks and "blocks." not in name:
            continue
        if not name.endswith(targets):
            continue
        to_replace.append((name, module))

    for name, module in to_replace:
        parent, attr = _get_parent_module(model, name)
        setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha, dropout=dropout, freeze_base=True))

    return len(to_replace)


def mark_only_lora_as_trainable(
        model: nn.Module,
        train_font_emb: bool = False,
        train_content_encoder: bool = False,
        train_style_encoder: bool = False,
):
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            if module.lora_A is not None:
                module.lora_A.requires_grad = True
            if module.lora_B is not None:
                module.lora_B.requires_grad = True
    if train_font_emb:
        model.net.y_embedder.font_embedding.weight.requires_grad = True
    if train_content_encoder:
        for param in model.net.y_embedder.content_encoder.parameters():
            param.requires_grad = True
    if train_style_encoder:
        for param in model.net.y_embedder.style_encoder.parameters():
            param.requires_grad = True


def count_trainable_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_checkpoint_path(path: str):
    if path is None:
        return None
    if os.path.isdir(path):
        latest_path = os.path.join(path, "checkpoint-latest.pth")
        if os.path.exists(latest_path):
            return latest_path
        return os.path.join(path, "checkpoint-last.pth")
    return path


def _is_lora_state_dict(state_dict):
    return any('.base.weight' in k for k in state_dict)


def _adapt_font_embedding_weight(
        state_dict: dict[str, torch.Tensor],
        model_state_dict: dict[str, torch.Tensor],
):
    key = "net.y_embedder.font_embedding.weight"
    if key not in state_dict or key not in model_state_dict:
        return state_dict, []

    source = state_dict[key]
    target = model_state_dict[key]
    if source.shape == target.shape:
        return state_dict, []

    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise RuntimeError(
            "Unsupported font embedding shape mismatch: "
            f"checkpoint={tuple(source.shape)} model={tuple(target.shape)}"
        )

    adapted = dict(state_dict)
    resized = target.clone()
    copy_rows = min(max(source.shape[0] - 1, 0), max(target.shape[0] - 1, 0))
    if copy_rows > 0:
        resized[:copy_rows] = source[:copy_rows]
    resized[-1] = source[-1]
    adapted[key] = resized

    messages = [
        "Adapted font embedding from "
        f"{tuple(source.shape)} to {tuple(target.shape)} by copying {copy_rows} font rows "
        "and preserving the checkpoint's unconditional row."
    ]
    if target.shape[0] - 1 > copy_rows:
        messages.append(
            f"Initialized {target.shape[0] - 1 - copy_rows} extra font rows from the current model init."
        )
    return adapted, messages


def load_state_dict_with_font_embedding_resize(
        model: nn.Module,
        state_dict: dict[str, torch.Tensor],
        *,
        strict: bool = True,
):
    model_state_dict = model.state_dict()
    adapted_state_dict, messages = _adapt_font_embedding_weight(state_dict, model_state_dict)

    shape_mismatches = []
    for key, value in adapted_state_dict.items():
        if key in model_state_dict and model_state_dict[key].shape != value.shape:
            shape_mismatches.append(
                f"{key}: checkpoint={tuple(value.shape)} model={tuple(model_state_dict[key].shape)}"
            )
    if shape_mismatches:
        raise RuntimeError(
            "Checkpoint still has incompatible parameter shapes after font embedding adaptation:\n"
            + "\n".join(shape_mismatches)
        )

    load_result = model.load_state_dict(adapted_state_dict, strict=strict)
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    if strict and (missing_keys or unexpected_keys):
        details = []
        if missing_keys:
            details.append(f"missing={missing_keys}")
        if unexpected_keys:
            details.append(f"unexpected={unexpected_keys}")
        raise RuntimeError("Strict checkpoint load failed: " + "; ".join(details))

    return messages


def add_lora_args(parser: argparse.ArgumentParser):
    parser.add_argument("--base_checkpoint", default="", type=str,
                        help="Path to a full-precision checkpoint (file or folder) to initialize weights.")
    parser.add_argument("--lora_r", default=8, type=int, help="LoRA rank.")
    parser.add_argument("--lora_alpha", default=16, type=int, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", default=0.0, type=float, help="LoRA dropout.")
    parser.add_argument("--lora_targets", default="qkv,proj,w12,w3", type=str,
                        help="Comma-separated list of Linear suffixes to LoRA-wrap.")
    parser.add_argument(
        "--train_content_encoder",
        action="store_true",
        help="Also finetune the content encoder alongside LoRA weights. This increases VRAM usage.",
    )
    parser.add_argument(
        "--train_style_encoder",
        action="store_true",
        help="Also finetune the style encoder alongside LoRA weights. This increases VRAM usage.",
    )
    return parser
