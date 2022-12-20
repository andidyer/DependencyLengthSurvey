import random
import argparse
from pathlib import Path
from src.load_treebank import TreebankLoader
from src.sentence_permuter import SentencePermuter
from src.treebank_processor import TreebankProcessor, FileProcessor
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
            "optimal_projective",
            "optimal_projective_weight",
            "original_order",
        ),
        help="The type of permutation to perform",
    )

    parser.add_argument(
        "--outfile", type=Path, help="The file to output the permuted treebank(s) to"
    )

    parser.add_argument(
        "--min_len",
        type=int,
        default=1,
        help="Exclude sentences with less than a given minimum number of tokens",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=999,
        help="Exclude sentences with more than a given maximum number of tokens",
    )

    parser.add_argument("--verbose", action="store_true", help="Verbosity")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.random_seed)

    # Make loader with cleaner
    loader = TreebankLoader(remove_fields={"deprel": "punct"}, min_len=args.min_len, max_len=args.max_len)

    # Make permuter
    permuter = SentencePermuter(args.permutation_mode)

    # Make treebank processor
    treebank_processor = TreebankProcessor(permuter)

    # Make file processor
    file_processor = FileProcessor(loader, treebank_processor)

    # Process and dump to file
    file_processor.process_file(args.treebank, args.outfile)


if __name__ == "__main__":
    main()
