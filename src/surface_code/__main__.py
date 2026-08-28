import argparse
import random

from .code import RotatedSurfaceCode


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a rotated surface code")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    code = RotatedSurfaceCode(args.distance)
    rng = random.Random(args.seed)
    detected = 0
    for _ in range(args.shots):
        error = code.sample_depolarizing_error(args.error_rate, rng)
        detected += any(code.syndrome(error))
    print(f"distance={args.distance} shots={args.shots} detected={detected} rate={detected / args.shots:.6f}")


if __name__ == "__main__":
    main()

