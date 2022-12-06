import random
import argparse
from pathlib import Path
from src.load_treebank import TreebankLoader


def parse_args():
    parser = argparse.ArgumentParser()

    required = parser.add_argument_group("required arguments")

    treebank_source = required.add_mutually_exclusive_group()
    treebank_source.add_argument(
        "--treebank", type=Path, help="Treebank to load and permute"
    )
    treebank_source.add_argument(
        "--directory", type=Path, help="Directory to load and permute"
    )

    parser.add_argument(
        "--random_seed", type=int, default=0, help="Random seed for permutation"
    )
    parser.add_argument(
        "--permutation_mode",
        type=str,
        choices=(
            "random_nonprojective",
            "random_projective",
            "random_projective_fixed",
            "random_same_valency",
            "random_same_side",
        ),
        help="The type of permutation to perform",
    )

    parser.add_argument(
        "output_file", type=Path, help="The file to output the permuted treebank(s) to"
    )

    parser.add_argument("--verbose", action="store_true", help="Verbosity")

    args = parser.parse_args()

    return args


def main():
    pass


if __name__ == "__main__":
    main()
