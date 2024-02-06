import random

from src.model_treebanks import HeadDirectionEntropyModeller
from src.sentence_cleaner import SentenceCleaner
from src.load_treebank import TreebankLoader

import logging
import json
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()

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
    "--deprels",
    type=Path,
    help="Path to tzt files with deprels",
)
parser.add_argument(
    "--callable_loading",
    action="store_true",
    help="Load the treebanks afresh on each epoch as a callable function. Recommended only for reading very large data",
)
parser.add_argument(
    "--log_level",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default="INFO",
    help="Set the logging level (default: INFO)",
)

args = parser.parse_args()

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=args.log_level
)

# Get list of deprels
with open(args.deprels, encoding="utf-8") as fin:
    DEPRELS = list(line.strip() for line in fin if line.strip() != "")

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
    dev_sentences = lambda: loader.iter_load_glob(args.dev_directory, args.dev_glob)

else:
    train_sentences = list(loader.iter_load_glob(args.train_directory, args.train_glob))
    dev_sentences = list(loader.iter_load_glob(args.dev_directory, args.dev_glob))


# Make modeller
logging.info("Making conditional entropy modeller")
modeller = HeadDirectionEntropyModeller()

# Fit model to training set
logging.info("Fitting model to training set")
modeller.fit(train_sentences)

# Predict training set conditional entropy
training_entropy = modeller.predict(train_sentences)
logging.info(f"Training set entropy: {training_entropy}")

# Predict development set conditional entropy
dev_entropy = modeller.predict(dev_sentences)
logging.info(f"Development set entropy: {dev_entropy}")

# Get model entropy
model_entropy = modeller.model_entropy()
logging.info(f"Model entropy: {model_entropy}")

# Get per label entropy
per_label_entropy = modeller.model_label_entropies()
logging.info(f"Per label entropy: {json.dumps(per_label_entropy, indent=1)}")

fout = open(args.output_file, "w", encoding="utf-8")

json.dump(
    {
        "model_conditional_entropy": 0.2,
        "training_set_conditional_entropy": training_entropy,
        "development_set_conditional_entropy": dev_entropy,
        "model_per_label_entropy": per_label_entropy,
     },
    fout,
    indent=1,
          )

fout.close()
