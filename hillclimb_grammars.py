import random

from src.grammar_hillclimb import (
    GrammarHillclimb,
    AnalyzerFactory
)
from src.sentence_cleaner import SentenceCleaner
from src.load_treebank import TreebankLoader
from src.utils.fileutils import load_ndjson

import logging
import json
from dataclasses import asdict
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
    "--objectives",
    type=str,
    nargs="+",
    choices=["DependencyLength", "IntervenerComplexity"],
    help="Choose which analysis modes to use"
)
parser.add_argument(
    "--weights",
    type=int,
    nargs="+",
    help="Weight to assign to each analyzer for optimisation"
)
parser.add_argument(
    "--deprels",
    type=Path,
    help="Path to tzt files with deprels",
)
parser.add_argument(
    "--baseline_grammar",
    type=Path,
    nargs="?",
    help="Path to baseline grammar",
)
parser.add_argument(
    "--epochs",
    type=int,
    help="Number of epochs to train",
)
parser.add_argument(
    "--burnin",
    type=int,
    default=0,
    help="Number of epochs to burn-in (i.e. train without output)",
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
parser.add_argument(
    "--random_seed",
    type=int,
    default=1,
)

args = parser.parse_args()

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=args.log_level
)

BURNIN = args.burnin
EPOCHS = args.epochs

random.seed(args.random_seed)

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

# Load baseline grammar if provided
if args.baseline_grammar is not None:
    logging.info(f"Loading baseline grammar from {args.baseline_grammar}")
    baseline_grammar = list(load_ndjson(args.baseline_grammar))[0]
else:
    baseline_grammar = None


logging.info("Making analyzers")
analyzers = AnalyzerFactory.create_analyzers(args.objectives)

logging.info("Making analyzer weights")
weights = args.weights

assert len(analyzers) == len(weights), "Number of analyzers and weights does not match"

logging.info("Instantiating grammar hillclimb")
mcmc = GrammarHillclimb(
    deprels=DEPRELS, analyzers=analyzers, objective_weights=weights
)

logging.info("Beginning grammar generation")
grammars_generator = mcmc.train_grammars(
    train_sentences,
    dev_sentences,
    baseline_grammar=baseline_grammar,
    burnin=BURNIN,
    epochs=EPOCHS,
)

fout = open(args.output_file, "w", encoding="utf-8")

for itemi in grammars_generator:
    json_itemi = asdict(itemi)
    print(json.dumps(json_itemi), file=fout)

fout.close()
