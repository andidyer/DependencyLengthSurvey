import random
import argparse
from pathlib import Path
from src.load_treebank import TreebankLoader
from src.treebank_permutation import TreebankPermuter
from src.utils.fileutils import FileDumper


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
            "optimal_projective"
        ),
        help="The type of permutation to perform",
    )

    parser.add_argument(
        "--outfile", type=Path, help="The file to output the permuted treebank(s) to"
    )

    parser.add_argument("--verbose", action="store_true", help="Verbosity")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.random_seed)

    # Make loader
    loader = TreebankLoader(remove_fields={"deprel": "punct"})

    # Load treebank
    treebank = loader.load_treebank(args.treebank)

    # Make permuter
    permuter = TreebankPermuter(args.permutation_mode)

    # Perform treebank permutation
    treebank_stream = permuter.yield_permute_treebank(treebank)

    # Dump to file
    FileDumper.dump_treebank_as_conllu(args.outfile, treebank_stream)


if __name__ == "__main__":
    main()
