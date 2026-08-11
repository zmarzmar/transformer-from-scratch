# Transformer from Scratch

Building a GPT-style language model from the ground up, one component at a time,
then measuring what each part actually contributes.

Study repo — everything is written by hand rather than imported, and every stage
ends with a number instead of "it works".


## Progress

| Stage | Topic | Status |
|-------|-------|--------|
| 0 | micrograd — scalar autograd engine | in progress |
| 1 | BPE tokenizer | |
| 2 | bigram → MLP | |
| 3 | attention | |
| 4 | GPT + one-at-a-time ablations | |


## Structure

```
docs/notes/    study notes (Korean)
results/       loss curves, csv
```


## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib ipykernel
```

Open `micrograd.ipynb` and select the `.venv` kernel.


## Notes

[`docs/notes/`](docs/notes/) — written in Korean.


## Reference

Following [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
by Andrej Karpathy.
