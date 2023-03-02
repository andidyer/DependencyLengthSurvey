import logging
from abc import ABC
from pathlib import Path

from src.file_dumper import FileDumper
from src.load_treebank import TreebankLoader
from src.treebank_processor import TreebankProcessor


class FileProcessor(ABC):
    """ "
    Takes in a file or set of files and processes them with the following steps:
    - loading and cleaning
    - analysis OR permutation
    - dumping to file(s)
    """

    def __init__(
        self, loader: TreebankLoader, processor: TreebankProcessor, dumper: FileDumper
    ):
        self.loader = loader
        self.processor = processor
        self.dumper = dumper

    def load_conllu_file(self, infile: Path):
        return self.loader.load_treebank(infile)

    def process_file(self, infile: Path, outfile: Path):
        # Override this
        treebank = self.loader.load_treebank(infile)
        processed_data = self.processor.process_treebank(treebank)
        self.dumper.write_to_file(processed_data, outfile)

    def process_glob(self, indir: Path, glob_pattern: str, outdir: Path):
        # Get list of infiles to process
        indir_path = Path(indir)
        infiles = indir_path.glob(glob_pattern)

        for infile in infiles:
            logging.info(f"Processing file: {infile}")

            outfile = self.dumper.make_equivalent_paths(indir, infile, outdir)

            self.process_file(infile, outfile)
