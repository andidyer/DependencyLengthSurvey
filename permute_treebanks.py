import argparse
import random
from pathlib import Path
import logging

from src.file_processor import FileProcessor, FilePermuter
from src.load_treebank import TreebankLoader
from src.sentence_permuter import RandomProjectivePermuter, RandomSameValencyPermuter, RandomSameSidePermuter, FixedOrderPermuter, OptimalProjectivePermuter, SentencePermuter
from src.treebank_processor import TreebankPermuter
from src.utils.fileutils import load_ndjson
from src.utils.processor_factories import treebank_permuter_factory


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
            "random_projective",
            "random_same_valency",
            "random_same_side",
            "optimal_projective",
            "original_order",
            "fixed_order"
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
        "--fields_to_remove",
        type=str,
        nargs="*",
        choices=["form", "lemma", "upos", "xpos", "feats", "deps", "misc"],
        help="Masks any fields in a conllu that are not necessary; can save some space"
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
        "--mask_words",
        action="store_true",
        help="Mask all words in the treebank. Token forms and lemma will be represented only by original token index."
    )

    optional.add_argument("--verbose", action="store_true", help="Verbosity")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Set random seed
    random.seed(args.random_seed)

    # Set logging level to info if verbose
    if args.verbose:
        level=logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(format= "%(asctime)s %(levelname)s %(message)s", level=level)

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
        mask_words=args.mask_words,
    )

    # Make treebank processors
    treebank_processors = []

    if args.n_times:

        for i in range(args.n_times):
            processor = treebank_permuter_factory(args.permutation_mode)
            treebank_processors.append(processor)

    elif args.grammars:
        grammars = load_ndjson(args.grammars)
        logging.info(f"Instantiating {len(grammars)} processors of permuter type {args.permutation_mode}")
        if not args.permutation_mode == "fixed_order":
            logging.warning(f"Grammars are only compatible with a fixed_order permuter. Attempting to use it with any other type may cause errors and is definitely a waste of compute.")
        for grammar in grammars:
            processor = treebank_permuter_factory(args.permutation_mode, grammar)
            treebank_processors.append(processor)

    else:
        logging.info(f"Instantiating single processor of permuter type {args.permutation_mode}")
        processor = treebank_permuter_factory(args.permutation_mode)
        treebank_processors.append(processor)

    # Make file processor
    file_processor = FilePermuter(loader, treebank_processors)

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
