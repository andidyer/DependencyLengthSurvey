import argparse
from pathlib import Path
from src.dependency_lengths import (
    TreebankDependencyLengthChecker,
)
from src.sentence_cleaner import SentenceCleaner
from src.statistics import TreebankStatistics
from src.load_treebank import TreebankLoader

"""

"""

def parse_args():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group("required arguments")
    treebank_source = required.add_mutually_exclusive_group()
    treebank_source.add_argument("--treebank", help="Treebank to load and analyse")
    treebank_source.add_argument("--directory", help="Directory to load and analyse")
    required.add_argument(
        "--datafile",
        type=Path,
        help="File to output the per-sentence statistics json to",
    )
    parser.add_argument("--count_root", action="store_true",
                        help="Whether to count the dependency length of the root node")

    parser.add_argument("--min_len", type=int, default=1, help="Exclude sentences with less than a given minimum number of tokens")
    parser.add_argument("--max_len", type=int, default=999, help="Exclude sentences with more than a given maximum number of tokens")

    return parser.parse_args()


def main():
    args = parse_args()

    # Make cleaner
    cleaner = SentenceCleaner({"deprel": "punct"})

    # Make loader
    loader = TreebankLoader(cleaner, min_len=args.min_len, max_len=args.max_len)

    if args.treebank is not None:
        treebank = loader.iter_load_treebank(args.treebank)
    elif args.directory is not None:
        treebank = loader.iter_load_directory(args.directory)
    checker = TreebankDependencyLengthChecker(count_root=args.count_root)
    sentence_data = checker.yield_treebank_sentence_data(treebank)
    TreebankStatistics.dump_treebank_data_as_ndjson(sentence_data, args.datafile)


if __name__ == "__main__":
    main()
