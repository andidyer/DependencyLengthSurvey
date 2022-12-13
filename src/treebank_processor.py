from pathlib import Path
from typing import List, Iterator
from conllu import SentenceList
from src.load_treebank import TreebankLoader
from src.utils.fileutils import FileDumper
from typing import List, Union, Callable
from pathlib import Path


class Handler:
    """"
    Class for performing an operation on many files,
    keeping track of the files as it does and directing
    the output to the correct structure.

    Basically an IO handler.
    """

    @staticmethod
    def perform_action(inner_function: Callable, infiles: List[Path]):
        """"Reads a set of conllu files and executes the function on them."""
        for infile in infiles:
            inner_function(infile, **kwargs)
