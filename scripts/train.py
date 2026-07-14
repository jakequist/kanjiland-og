"""Training entry point (M3). Skeleton only — built during M3.

Usage: uv run python scripts/train.py --config configs/<name>.yaml
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise NotImplementedError("M3: build with Claude Code")


if __name__ == "__main__":
    main()
