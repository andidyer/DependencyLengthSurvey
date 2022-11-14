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
    treebank_source = parser.add_mutually_exclusive_group()
    treebank_source.add_argument("--treebank", help="Treebank to load and analyse")
    treebank_source.add_argument("--directory", help="Directory to load and analyse")
    parser.add_argument(
        "--datafile",
        type=Path,
        help="File to output the per-sentence statistics json to",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cleaner = SentenceCleaner(upos="PUNCT")
    loader = TreebankLoader(cleaner)
    if args.treebank is not None:
        treebank = loader.iter_load_treebank(args.treebank)
    elif args.directory is not None:
        treebank = loader.iter_load_directory(args.directory)
    checker = TreebankDependencyLengthChecker(ignore_root=True)
    sentence_data = checker.yield_treebank_sentence_data(treebank)
    TreebankStatistics.dump_treebank_data_as_json(sentence_data, args.datafile)


if __name__ == "__main__":
    main()
