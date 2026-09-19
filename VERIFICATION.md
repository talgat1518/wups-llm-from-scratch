# Verification record

This repository was smoke-tested before publication in a clean CPU environment.

## Automated tests

```text
2 passed in 2.84s
```

The tests verify UTF-8 tokenizer round-tripping, model forward pass, finite loss and autoregressive generation.

## Training verification run

Command:

```bash
python train.py --data data/sample.txt --steps 400 --batch-size 8 --block-size 64 --d-model 64 --n-layers 2 --n-heads 4 --out checkpoints/demo.pt
```

Observed output:

```text
device=cpu parameters=119,424 bytes_in_corpus=1,542
step=0001 train_loss=5.5512
step=0100 train_loss=3.3406
step=0200 train_loss=2.5679
step=0300 train_loss=1.9536
step=0400 train_loss=1.9078
validation_loss=3.0499
```

Checkpoint SHA-256 from that run:

```text
0fac362434a71603a63ccc5aca7bf29fb820f1584d4b780685383ef2fd7d2c3c
```

The checkpoint binary is intentionally not committed. It can be reproduced from the public code, corpus and command above. The hash is stored in `artifacts/demo_checkpoint.sha256`.

## Generation smoke test

The saved checkpoint was loaded by `generate.py` and produced autoregressive output from a text prompt. The tiny model is not expected to produce high-quality language; the purpose of this run is to verify the full path from random initialization to saved learned weights and inference.

## Interpretation

The decreasing training loss confirms that randomly initialized parameters were updated by the training loop. This is a small engineering proof-of-concept, not a production-scale language model.
