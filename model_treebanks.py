import random

from src.model_treebanks import HeadDirectionEntropyModeller, BigramMutualInformationModeller
from src.sentence_cleaner import SentenceCleaner
from src.load_treebank import TreebankLoader

import logging
import json
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--modeller",
    type=str,
    choices=["HeadDirectionEntropy", "BigramMutualInformation"]
)
parser.add_argument(
    "--train_directory",
    type=Path,
    default=".",
    help="Directory from which to find training treebanks by globbing",
)
parser.add_argument(
    "--dev_directory",
    type=Path,
    nargs="?",
    help="Directory from which to find dev treebanks by globbing",
)
parser.add_argument(
    "--train_glob",
    type=str,
    help="glob pattern for recursively finding files that match the pattern",
)
parser.add_argument(
    "--dev_glob",
    type=str,
    nargs="?",
    help="glob pattern for recursively finding files that match the pattern",
)
parser.add_argument(
    "--output_file",
    type=str,
    help="ndjson file to output the training outputs to",
)
parser.add_argument(
    "--callable_loading",
    action="store_true",
    help="Load the treebanks afresh on each epoch as a callable function. Recommended only for reading very large data",
)
parser.add_argument(
    "--log_level",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level (default: INFO)",
)
parser.add_argument(
    "--lowercase",
    action="store_true",
    help="Lowercase token forms",
)
parser.add_argument(
    "--threshold",
    type=int,
    default=1,
    help="Threshold for bigram occurrences"
)
parser.add_argument(
    "--normalized",
    action="store_true",
    help="Normalize MI score",
)

args = parser.parse_args()

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=args.log_level
)

# Make cleaner
logging.info("Making sentence cleaner")
cleaner = SentenceCleaner(remove_config=[{"deprel": "punct"}])

# Make loader
logging.info("Making sentence loader")
loader = TreebankLoader(cleaner=cleaner)

logging.info("Loading sentences")
if args.callable_loading:
    train_sentences = lambda: loader.iter_load_glob(
        args.train_directory, args.train_glob
    )
    if all((args.dev_directory, args.dev_glob)):
        dev_sentences = lambda: loader.iter_load_glob(args.dev_directory, args.dev_glob)

else:
    train_sentences = list(loader.iter_load_glob(args.train_directory, args.train_glob))
    if all((args.dev_directory, args.dev_glob)):
        dev_sentences = list(loader.iter_load_glob(args.dev_directory, args.dev_glob))

modeller_choices = {
    "HeadDirectionEntropy": HeadDirectionEntropyModeller,
    "BigramMutualInformation": BigramMutualInformationModeller,
}

optionals = {
    "lower": args.lowercase,
    "threshold": args.threshold,
    "normalized": args.normalized,
}

# Make modeller
logging.info("Making conditional entropy modeller")
modeller_class = modeller_choices[args.modeller]
modeller = modeller_class(**optionals)

readout_scores = {}

# Fit model to training set
logging.info("Fitting model to training set")
modeller.fit(train_sentences)

# Get model score
model_score = modeller.model_score()
logging.info(f"Model score: {model_score}")
readout_scores["model_score"] = model_score

fout = open(args.output_file, "w", encoding="utf-8")

json.dump(
    readout_scores,
    fout,
    indent=1,
          )

fout.close()
