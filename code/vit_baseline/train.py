"""Train a seeded ViT on CUDA with the existing v2 supervision protocol."""

from __future__ import annotations

from climatology_baseline.runtime import make_parser, train_neural

from vit_baseline.model import ViTReconstructor


def main() -> None:
    """Parse ViT settings and seed the shared CUDA trainer before initialization."""
    parser = make_parser("vit_baseline")
    parser.set_defaults(patch_size=3)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()
    train_neural(args, lambda: ViTReconstructor(args.embedding_dim, args.blocks, args.heads, args.dropout))


if __name__ == "__main__":
    main()
