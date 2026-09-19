"""Generate text using a checkpoint produced by train.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import ModelConfig, WupsGPT
from tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/wups_gpt.pt")
    p.add_argument("--prompt", default="AI ")
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=1518)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    ckpt = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = WupsGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [10]
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    out = model.generate(
        idx,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
