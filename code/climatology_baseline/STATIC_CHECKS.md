# Static delivery checks

2026-11-19: ViT added and neural training defaults changed to 250 epochs.
No CPU/GPU training, fitting, model forward pass or dataset evaluation was performed.

- Ruff 0.15.1 lint and format checks passed for 18 Python files in the five baseline packages.
- mypy 1.20.2 strict checks passed for those 18 files using server Python 3.11 and installed PyTorch type information.
- Source compilation passed for 20 Python files, including the two Swin trainers; code was not executed.
- Public annotations and docstrings were checked with AST.
- bash -n passed for 16 shell scripts.
- AST checks confirmed epochs=250 in all three neural Python argument parsers.
- Shell entry defaults were checked as 250; statistical references retain no iterative epochs.
- ViT is registered in checkpoint evaluation, CUDA launch dispatch and stochastic seed summaries.
- Legacy reader/loss/evaluator interfaces remain outside the new strict typing scope.
- CUDA behavior, numerical results and GPU memory use remain untested.
- Seeded best-effort determinism does not guarantee deterministic CUDA grid_sample backward.
