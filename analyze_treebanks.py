import argparse
import random
from pathlib import Path

from src.file_processor import FileAnalyzer
from src.load_treebank import TreebankLoader
from src.utils.fileutils import load_ndjson
from src.utils.processor_factories import treebank_analyzer_factory


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
        "--fields_to_remove",
        type=str,
        nargs="*",
        choices=["form", "lemma", "upos", "xpos", "feats", "deps", "misc"],
        help="Masks any fields in a conllu that are not necessary; can save some space",
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
        "--count_root",
        action="store_true",
        help="Include the root node in the sentence analysis",
    )

    optional.add_argument("--verbose", action="store_true", help="Verbosity")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.random_seed)

    # Loads the remove config (for removing tokens of given type) or ignores
    if args.remove_config:
        remove_config = load_ndjson(args.remove_config)
    else:
        remove_config = None

    # Make loader with cleaner
    loader = TreebankLoader(
        remove_config=remove_config,
        fields_to_remove=args.fields_to_remove,
        min_len=args.min_len,
        max_len=args.max_len,
    )

    # Make treebank analyzer

    analyzer = treebank_analyzer_factory(args.count_root)

    # Make file processor
    file_processor = FileAnalyzer(loader, analyzer)

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
