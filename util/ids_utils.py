from __future__ import annotations

from pathlib import Path

import torch


def _split_ids_column(text: str) -> list[str]:
    sequences = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        if "(" in item:
            item = item.split("(", 1)[0].strip()
        if item:
            sequences.append(item)
    return sequences


def load_ids_mapping(ids_path: str | Path) -> dict[int, str]:
    path = Path(ids_path)
    if not path.is_file():
        raise FileNotFoundError(f"IDS file not found: {path}")

    mapping: dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split(maxsplit=2)
            if len(parts) < 2:
                continue

            char = parts[0].strip()
            if len(char) != 1:
                continue

            primary = _split_ids_column(parts[1])
            alternative = _split_ids_column(parts[2]) if len(parts) > 2 else []
            sequence = primary[0] if primary else (alternative[0] if alternative else "")
            if sequence:
                mapping[ord(char)] = sequence

    return mapping


def build_ids_resources(unicode_codepoints: list[int], ids_path: str | Path) -> dict:
    ids_mapping = load_ids_mapping(ids_path)
    sequences = [ids_mapping.get(int(codepoint), "") for codepoint in unicode_codepoints]
    token_vocab = sorted({token for sequence in sequences for token in sequence})
    token_to_id = {token: idx + 1 for idx, token in enumerate(token_vocab)}
    max_len = max((len(sequence) for sequence in sequences), default=0)

    token_ids = torch.zeros(len(unicode_codepoints) + 1, max_len, dtype=torch.long)
    token_mask = torch.zeros(len(unicode_codepoints) + 1, max_len, dtype=torch.float32)
    missing = []

    for idx, sequence in enumerate(sequences):
        if not sequence:
            missing.append(int(unicode_codepoints[idx]))
            continue
        ids = [token_to_id[token] for token in sequence]
        token_ids[idx, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        token_mask[idx, :len(ids)] = 1.0

    return {
        "ids_mapping": ids_mapping,
        "ids_vocab": token_vocab,
        "ids_vocab_size": len(token_vocab),
        "ids_max_len": max_len,
        "ids_token_ids": token_ids,
        "ids_token_mask": token_mask,
        "missing_codepoints": missing,
    }
