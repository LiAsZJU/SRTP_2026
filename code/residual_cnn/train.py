"""Train the residual CNN on CUDA with the frozen v2 protocol."""

from __future__ import annotations

from climatology_baseline.runtime import make_parser, train_neural

from residual_cnn.model import ResidualCNN


def main() -> None:
    """Parse CNN settings and start a fresh CUDA-only experiment."""
    parser = make_parser("residual_cnn")
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=4)
    args = parser.parse_args()
    train_neural(args, lambda: ResidualCNN(args.width, args.blocks))


if __name__ == "__main__":
    main()
