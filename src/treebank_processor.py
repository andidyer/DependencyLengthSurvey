import logging
from typing import List, Dict, Union
from abc import ABC, abstractmethod

from conllu import TokenList, SentenceList

from src.utils.abstractclasses import SentenceMainProcessor


class TreebankProcessor(ABC):
    """ "
    Class for performing an operation on a treebank.
    """

    def __init__(self, sentence_processor: SentenceMainProcessor):
        self.sentence_processor = sentence_processor

    @abstractmethod
    def process_treebank(self, treebank: SentenceList, **kwargs):
        pass


class TreebankPermuter(TreebankProcessor):
    def __init__(self, sentence_processor: SentenceMainProcessor):
        super().__init__(sentence_processor)
        self.fileext = ".conllu"

    def process_treebank(self, treebank: SentenceList, **kwargs):
        processed_sentences = []
        for sentence in treebank:
            processed_sentences.append(
                self.sentence_processor.process_sentence(sentence, **kwargs)
            )

        return processed_sentences


class TreebankAnalyzer(TreebankProcessor):
    def __init__(self, sentence_processor: SentenceMainProcessor):
        super().__init__(sentence_processor)
        self.fileext = ".ndjson"

    def process_treebank(self, treebank: SentenceList, **kwargs):
        processed_sentences = []
        for sentence in treebank:
            processed_sentences.append(
                self.sentence_processor.process_sentence(sentence, **kwargs)
            )

        return processed_sentences
