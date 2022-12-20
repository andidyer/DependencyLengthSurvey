import argparse
from pathlib import Path

from src.sentence_analyzer import SentenceAnalyzer
from src.treebank_processor import TreebankProcessor, FileProcessor
from src.load_treebank import TreebankLoader


def parse_args():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument("--treebank", help="Treebank to load and analyse")
    required.add_argument(
        "--outfile",
        type=Path,
        help="File to output the per-sentence statistics json to",
    )
    optional.add_argument(
        "--count_root",
        action="store_true",
        help="Whether to count the dependency length of the root node",
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

    return parser.parse_args()


def main():
    args = parse_args()

    # Make loader with cleaner
    loader = TreebankLoader(
        remove_fields={"deprel": "punct"}, min_len=args.min_len, max_len=args.max_len
    )

    # Make sentence analyzer
    analyzer = SentenceAnalyzer(count_root=args.count_root)

    # Make treebank processor with analyzer
    treebank_processor = TreebankProcessor(analyzer)

    # Make file processor
    file_processor = FileProcessor(loader, treebank_processor)

    # Process and dump to file
    file_processor.process_file(args.treebank, args.outfile)


if __name__ == "__main__":
    main()
