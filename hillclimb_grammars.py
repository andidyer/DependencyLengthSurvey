from src.grammar_hillclimb import GrammarHillclimb, DLAnalyzer, IntervenerComplexityAnalyzer
from src.sentence_cleaner import SentenceCleaner
from src.load_treebank import TreebankLoader

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
    default=".",
    help="Directory from which to find dev treebanks by globbing",
)
parser.add_argument(
    "--train_glob",
    type=str,
    default="*",
    help="glob pattern for recursively finding files that match the pattern",
)
parser.add_argument(
    "--dev_glob",
    type=str,
    default="*",
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
    help="Load the treebanks afresh on each epoch as a callable function. Recommended only for reading very large data"
)
parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

args = parser.parse_args()

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=args.log_level)

BURNIN = args.burnin
EPOCHS = args.epochs

# Get list of deprels
with open(args.deprels, encoding="utf-8") as fin:
    DEPRELS = list(line.strip() for line in fin)

# Make cleaner
logging.info("Making sentence cleaner")
cleaner = SentenceCleaner(remove_config=[{"deprel": "punct"}])

# Make loader
logging.info("Making sentence loader")
loader = TreebankLoader(cleaner=cleaner)

logging.info("Loading sentences")
if args.callable_loading:
    train_sentences = lambda: loader.iter_load_glob(args.train_directory, args.train_glob)
    dev_sentences = lambda: loader.iter_load_glob(args.dev_directory, args.dev_glob)

else:
    train_sentences = list(loader.iter_load_glob(args.train_directory, args.train_glob))
    dev_sentences = list(loader.iter_load_glob(args.dev_directory, args.dev_glob))

logging.info("Making analyzers")
dl_analyzer = DLAnalyzer()
icm_analyzer = IntervenerComplexityAnalyzer()

logging.info("Instantiating grammar hillclimb")
mcmc = GrammarHillclimb(deprels=DEPRELS, analyzers=[dl_analyzer, icm_analyzer])

logging.info("Beginning grammar generation")
grammars_generator = mcmc.train_grammars(
    train_sentences,
    dev_sentences,
    burnin=BURNIN,
    epochs=EPOCHS,
)

fout = open(args.output_file, "w", encoding="utf-8")

for itemi in grammars_generator:
    json_itemi = asdict(itemi)
    print(json.dumps(json_itemi), file=fout)

fout.close()
