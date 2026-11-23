"""Package the existing training-month climatology as a CUDA-scored reference."""

from __future__ import annotations

from typing import cast

import torch

from climatology_baseline.model import ClimatologyBaseline
from climatology_baseline.runtime import finish_statistical_fit, make_objective, make_parser, prepare_run


def main() -> None:
    """Save and validate the training-only background without iterative training."""
    parser = make_parser("climatology_baseline")
    parser.set_defaults(epochs=0, patience=0, amp=False)
    args = parser.parse_args()
    source, device, config = prepare_run(args)
    objective = make_objective(source, device, config)
    mean = cast(torch.Tensor, objective.mean)
    std = cast(torch.Tensor, objective.std)
    model = ClimatologyBaseline(-mean / std).to(device)
    finish_statistical_fit(
        model,
        source,
        device,
        config,
        {"fit_passes": 0, "fit_method": "reuse_training_only_monthly_background", "stochastic_model": False},
    )


if __name__ == "__main__":
    main()
