"""Train WUPS GPT from random initialization and save an original checkpoint."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random

import torch

from model import ModelConfig, WupsGPT
from tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a tiny decoder-only Transformer from scratch")
    p.add_argument("--data", default="data/sample.txt")
    p.add_argument("--out", default="checkpoints/wups_gpt.pt")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1518)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    tokenizer = ByteTokenizer()
    text = Path(args.data).read_text(encoding="utf-8")
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    minimum = args.block_size + 2
    if len(tokens) < minimum:
        raise SystemExit(f"Dataset is too small: {len(tokens)} tokens; need >= {minimum}")

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )
    model = WupsGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    split = max(args.block_size + 2, int(len(tokens) * 0.9))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    if len(val_tokens) < minimum:
        val_tokens = train_tokens

    def batch(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        high = len(source) - args.block_size - 1
        starts = torch.randint(0, high, (args.batch_size,))
        x = torch.stack([source[i : i + args.block_size] for i in starts])
        y = torch.stack([source[i + 1 : i + args.block_size + 1] for i in starts])
        return x.to(device), y.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"device={device} parameters={param_count:,} bytes_in_corpus={len(tokens):,}")

    model.train()
    first_loss = None
    last_loss = None
    for step in range(1, args.steps + 1):
        x, y = batch(train_tokens)
        _, loss = model(x, y)
        assert loss is not None
        if first_loss is None:
            first_loss = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.item())
        if step == 1 or step == args.steps or step % args.log_every == 0:
            print(f"step={step:04d} train_loss={last_loss:.4f}")

    model.eval()
    with torch.no_grad():
        x, y = batch(val_tokens)
        _, val_loss = model(x, y)
    assert val_loss is not None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "project": "wups-llm-from-scratch",
        "model_config": model.config_dict(),
        "model_state_dict": model.state_dict(),
        "training": {
            "steps": args.steps,
            "seed": args.seed,
            "first_loss": first_loss,
            "last_loss": last_loss,
            "validation_loss": float(val_loss.item()),
            "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        },
        "tokenizer": {"type": "utf8-byte", "vocab_size": tokenizer.vocab_size},
    }
    torch.save(checkpoint, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"validation_loss={float(val_loss.item()):.4f}")
    print(f"checkpoint={out} sha256={digest}")


if __name__ == "__main__":
    main()
