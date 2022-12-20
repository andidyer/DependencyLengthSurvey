import json
from pathlib import Path
from typing import List, Iterator, Iterable, Dict, Union
from conllu import Token, TokenList, SentenceList

from src.utils.abstractclasses import (
    SentenceProcessor,
    SentencePreProcessor,
    SentenceMainProcessor,
)
from src.load_treebank import TreebankLoader
from src.sentence_cleaner import SentenceCleaner
from src.utils.fileutils import FileDumper
from src.utils.abstractclasses import SentenceProcessor, SentenceMainProcessor
from pathlib import Path


class TreebankProcessor:
    """ "
    Class for performing an operation on a treebank.

    Uses an inner function, whici
    """

    def __init__(self, sentence_processor: SentenceMainProcessor):
        self.sentence_processor = sentence_processor

    def process_treebank(self, treebank: SentenceList, **kwargs):
        """ "Reads a set of conllu files and executes the function on them."""
        processed_sentences: List[Union[TokenList, Dict]] = []
        for sentence in treebank:
            processed_sentences.append(
                self.sentence_processor.process_sentence(sentence, **kwargs)
            )
        return processed_sentences
