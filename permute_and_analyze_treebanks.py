import argparse
import random
from pathlib import Path
import logging

from src.file_processor import FileProcessor
from src.load_treebank import TreebankLoader
from src.sentence_cleaner import SentenceCleaner
from src.file_dumper import FileDumper
from src.utils.fileutils import load_ndjson
from src.utils.processor_factories import (
    treebank_permuter_factory,
    sentence_analyzer_factory,
    treebank_permuter_analyzer_factory,
)


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
            "RandomProjective",
            "RandomSameValency",
            "RandomSameSide",
            "OptimalOrder",
            "OriginalOrder",
            "FixedOrder",
        ),
        help="The type of permutation to perform",
    )

    required.add_argument(
        "--analysis_modes",
        type=str,
        nargs="+",
        choices=[
            "DependencyLength",
            "IntervenerComplexity",
            "SemanticSimilarity",
            "WordFrequency",
            "WordZipfFrequency",
        ],
        help="Metrics with which to analyse tokens/sentences",
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
        "--mask_words",
        action="store_true",
        help="Mask all words in the treebank. Token forms and lemma will be represented only by original token index.",
    )

    repetitions = optional.add_mutually_exclusive_group()

    repetitions.add_argument(
        "--n_times",
        type=int,
        help="Number of times to perform the permutation action on each treebank",
    )

    repetitions.add_argument(
        "--grammars",
        type=Path,
        help="Number of times to perform the permutation action on each treebank",
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

    optional.add_argument(
        "--language",
        type=str,
        help="Language (ISO639-1) for the WordFrequency analyzer",
    )

    optional.add_argument(
        "--aggregate",
        action="store_true",
        help="If true, token scores will be aggregated and the results for each sentence will be output in an ndjson",
    )
    optional.add_argument(
        "--verbosity",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Set the logging verbosity level (default: INFO)",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Set logging level according to verbosity
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=args.verbosity)

    # Set random seed
    random.seed(args.random_seed)

    # Loads the remove config (for removing tokens of given type) or ignores
    if args.remove_config:
        print(f"using remove config from {args.remove_config}")
        remove_config = load_ndjson(args.remove_config)
    else:
        remove_config = None

    # Make cleaner
    cleaner = SentenceCleaner(remove_config, args.fields_to_remove, args.mask_words)

    # Make loader
    loader = TreebankLoader(
        cleaner=cleaner,
        min_len=args.min_len,
        max_len=args.max_len,
    )

    if args.n_times:
        logging.info(
            f"Instantiating {args.n_times} processors of permuter type {args.permutation_mode}"
        )
        treebank_processor = treebank_permuter_analyzer_factory(
            args.permutation_mode,
            args.analysis_modes,
            n_times=args.n_times,
            count_root=args.count_root,
            aggregate=args.aggregate,
        )

    elif args.grammars:
        grammars = list(load_ndjson(args.grammars))

        logging.info(
            f"""
            Instantiating {len(grammars)} processors of permuter type {args.permutation_mode} 
            from grammars in {args.grammars}
            """
        )

        treebank_processor = treebank_permuter_analyzer_factory(
            args.permutation_mode,
            args.analysis_modes,
            grammars=grammars,
            count_root=args.count_root,
            aggregate=args.aggregate,
        )

    else:
        logging.info(
            f"Instantiating single processor of permuter type {args.permutation_mode}"
        )
        treebank_processor = treebank_permuter_analyzer_factory(args.permutation_mode)

    # Make file dumper
    extension = ".ndjson" if args.aggregate else ".conllu"
    file_dumper = FileDumper(extension=extension)

    # Make file processor
    file_processor = FileProcessor(loader, treebank_processor, file_dumper)

    # Handle input and output
    if args.treebank and args.outfile:

        # Process and dump to file
        file_processor.process_file(args.treebank, args.outfile)

    elif args.directory and args.outdir:

        # Get glob list of files and process them
        file_processor.process_glob(args.directory, args.glob_pattern, args.outdir)

    else:
        raise ValueError("Incorrect or incompatible use of input and output options.")


if __name__ == "__main__":
    main()
