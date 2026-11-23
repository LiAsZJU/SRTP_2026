"""Train the U-Net regressor on CUDA with the frozen v2 protocol."""

from __future__ import annotations

from climatology_baseline.runtime import make_parser, train_neural

from unet_baseline.model import UNetReconstructor


def main() -> None:
    """Parse U-Net settings and start a fresh CUDA-only experiment."""
    parser = make_parser("unet_baseline")
    parser.add_argument("--width", type=int, default=24)
    args = parser.parse_args()
    train_neural(args, lambda: UNetReconstructor(args.width))


if __name__ == "__main__":
    main()
