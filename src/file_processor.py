import copy
import json
import logging
from pathlib import Path
from typing import Union, Dict, Iterable, List, AnyStr, Any
from abc import ABC, abstractmethod

from conllu import TokenList, SentenceList

from src.load_treebank import TreebankLoader
from src.treebank_processor import TreebankProcessor, TreebankAnalyzer, TreebankPermuter


class FileProcessor(ABC):
    """ "
    Takes in a file or set of files and processes them with the following steps:
    - loading and cleaning
    - analysis OR permutation
    - dumping to file(s)
    """
    fileext = ".txt"

    def __init__(self, loader: TreebankLoader):
        self.loader = loader

    def load_conllu_file(self, infile: Path):
        return self.loader.load_treebank(infile)

    @abstractmethod
    def serialize_data(self, data_item: Any):
        """ "Utility function for serializing the data to a printable form
        :param data_item: a data item of whichever format
        :return: a printable form
        """
        pass

    def dump_to_file(self, data: Iterable, outfile: Path):
        with open(outfile, "w", encoding="utf-8") as fout:
            for data_item in data:
                serialized = self.serialize_data(data_item)
                print(serialized, file=fout)

    @abstractmethod
    def process_file(self, infile: Path, outfile: Path):
        # Override this
        pass

    def process_glob(self, indir: Path, glob_pattern: str, outdir: Path):
        """
        Does the IO process on a set of files specified by the glob rather than
        an individual file.

        Can only direct new files to a specified directory while preserving sub-
        directory structure.
        """
        indir_path = Path(indir)

        infiles = indir_path.glob(glob_pattern)
        for infile in infiles:
            logging.info(f"Processing file: {infile}")
            infile_relpath = Path(infile).relative_to(indir)
            parent = infile_relpath.parent
            stem = infile_relpath.stem
            extension = self.fileext
            outfile_parent = Path(outdir, parent)
            outfile_path = Path(outfile_parent, stem + extension)

            if not outfile_parent.exists():
                logging.info(f"Making parent path: {outfile_parent}")
                outfile_parent.mkdir(parents=True)

            self.process_file(infile, outfile_path)


class FilePermuter(FileProcessor):
    fileext = ".conllu"

    def __init__(self, loader: TreebankLoader, treebank_permuters: List[TreebankPermuter]):
        super().__init__(loader)
        self.treebank_permuters = treebank_permuters

    def process_file(self, infile: Path, outfile: Path):
        """Main function for processing a single file"""
        input_treebank = self.load_conllu_file(infile)
        processed_treebank = []
        for i, permuter in enumerate(self.treebank_permuters):
            input_treebank_copy = copy.deepcopy(input_treebank)
            processed = permuter.process_treebank(input_treebank_copy)
            processed_treebank.extend(processed)

        self.dump_to_file(processed_treebank, outfile)

    def serialize_data(self, data_item: TokenList):
        return data_item.serialize()


class FileAnalyzer(FileProcessor):
    fileext = ".ndjson"

    def __init__(self, loader: TreebankLoader, treebank_analyzer: TreebankAnalyzer):
        super().__init__(loader)
        self.treebank_analyzer = treebank_analyzer

    def process_file(self, infile: Path, outfile: Path):
        """Main function for processing a single file"""
        input_treebank = self.load_conllu_file(infile)
        processed_treebank = []

        input_treebank_copy = copy.deepcopy(input_treebank)
        processed = self.treebank_analyzer.process_treebank(input_treebank_copy)
        processed_treebank.extend(processed)

        self.dump_to_file(processed_treebank, outfile)

    def serialize_data(self, data_item: Dict):
        return json.dumps(data_item)


class FilePermuterAnalyzer(FileProcessor):
    """
    This class will take in a set of files, permute them, and analyze them in a single step.
    It will only output the final analysis ndjson, not the permuted treebank.
    Prefer to use this when the size of the permuted treebank files would otherwise be unreasonably large.
    """
    fileext = ".ndjson"

    def __init__(self, loader: TreebankLoader, treebank_permuters: List[TreebankPermuter], treebank_analyzer: TreebankAnalyzer):
        super().__init__(loader)
        self.treebank_permuters = treebank_permuters
        self.treebank_analyzer = treebank_analyzer

    def process_file(self, infile: Path, outfile: Path):
        """Main function for processing a single file"""
        input_treebank = self.load_conllu_file(infile)
        treebank_analysis_data: List[Dict] = []

        # For all provided permuters, permute the treebank and then analyze it with the analyzer
        for i, permuter in enumerate(self.treebank_permuters):
            input_treebank_copy = copy.deepcopy(input_treebank)
            permuted = permuter.process_treebank(input_treebank_copy)
            analysis: List[Dict] = self.treebank_analyzer.process_treebank(permuted)
            treebank_analysis_data.extend(analysis)

        self.dump_to_file(treebank_analysis_data, outfile)

    def serialize_data(self, data_item: Dict):
        return json.dumps(data_item)
