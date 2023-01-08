import argparse
import random
from pathlib import Path
import logging

from src.file_processor import FileProcessor
from src.load_treebank import TreebankLoader
from src.sentence_permuter import SentencePermuter
from src.treebank_processor import TreebankPermuter
from src.utils.fileutils import load_ndjson

logging.basicConfig(level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()

    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    treebank_source = required.add_mutually_exclusive_group()

    treebank_source.add_argument(
        "--treebank", type=Path, help="Treebank to load and permute"
    )

    treebank_source.add_argument(
        "--directory",
        type=Path,
        help="Directory from which to find treebanks by globbing",
    )

    optional.add_argument(
        "--glob_pattern",
        type=str,
        default="*",
        help="glob pattern for recursively finding files that match the pattern",
    )

    optional.add_argument(
        "--random_seed", type=int, default=0, help="Random seed for permutation"
    )
    required.add_argument(
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

    output = required.add_mutually_exclusive_group()

    output.add_argument(
        "--outfile", type=Path, help="The file to output the permuted treebank(s) to"
    )

    output.add_argument(
        "--outdir",
        type=Path,
        help="The directory to output the permuted treebank(s) to",
    )

    optional.add_argument(
        "--remove_config",
        type=Path,
        default=None,
        help="ndjson format list of token properties to exclude",
    )

    optional.add_argument(
        "--min_len",
        type=int,
        default=1,
        help="Exclude sentences with less than a given minimum number of tokens",
    )
    optional.add_argument(
        "--max_len",
        type=int,
        default=999,
        help="Exclude sentences with more than a given maximum number of tokens",
    )

    optional.add_argument(
        "--n_times",
        type=int,
        default=1,
        help="Number of times to perform the permutation action on each treebank",
    )

    optional.add_argument("--verbose", action="store_true", help="Verbosity")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.random_seed)

    if args.remove_config:
        remove_config = load_ndjson(args.remove_config)
    else:
        remove_config = None

    # Make loader with cleaner
    loader = TreebankLoader(
        remove_config=remove_config, min_len=args.min_len, max_len=args.max_len
    )

    # Make permuter
    permuter = SentencePermuter(args.permutation_mode)

    # Make treebank processor
    treebank_processor = TreebankPermuter(permuter, n_times=args.n_times)

    # Make file processor
    file_processor = FileProcessor(loader, treebank_processor)

    # Handle input and output
    if args.treebank and args.outfile:

        # Process and dump to file
        file_processor.process_file(args.treebank, args.outfile)

    elif args.directory and args.outdir:

        # Get glob list of files and process them
        file_processor.process_glob(args.directory, args.glob_pattern, args.outdir)

    else:
        raise argparse.ArgumentError(
            "Incorrect or incompatible use of input and output options."
        )


if __name__ == "__main__":
    main()
