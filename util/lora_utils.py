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


def count_trainable_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_checkpoint_path(path: str):
    if path is None:
        return None
    if os.path.isdir(path):
        return os.path.join(path, "checkpoint-last.pth")
    return path


def _is_lora_state_dict(state_dict):
    return any('.base.weight' in k for k in state_dict)


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
    return parser
