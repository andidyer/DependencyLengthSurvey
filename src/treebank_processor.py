import copy
from abc import ABC, abstractmethod
from typing import List

from conllu import SentenceList

from src.sentence_analyzer import SentenceAnalyzer
from src.sentence_permuter import SentencePermuter


class TreebankProcessor(ABC):
    """ "
    Class for performing an operation on a treebank.
    """

    def __init__(self):
        pass

    @abstractmethod
    def process_treebank(self, treebank: SentenceList, **kwargs):
        pass


class TreebankPermuter(TreebankProcessor):
    def __init__(self, sentence_permuters: List[SentencePermuter]):
        super().__init__()
        self.sentence_permuters = sentence_permuters

    def process_treebank(self, treebank: SentenceList, **kwargs):
        for permuter in self.sentence_permuters:
            new_treebank = copy.deepcopy(treebank)  # Avoids modifying previous output
            for sentence in new_treebank:
                yield permuter.process_sentence(sentence, **kwargs)


class TreebankAnalyzer(TreebankProcessor):
    def __init__(self, sentence_analyzer: SentenceAnalyzer):
        super().__init__()
        self.sentence_analyzer = sentence_analyzer

    def process_treebank(self, treebank: SentenceList, **kwargs):
        new_treebank = treebank.copy()
        for sentence in new_treebank:
            yield self.sentence_analyzer.process_sentence(sentence, **kwargs)


class TreebankPermuterAnalyzer(TreebankProcessor):
    def __init__(
        self,
        sentence_permuters: List[SentencePermuter],
        sentence_analyzer: SentenceAnalyzer,
    ):
        super().__init__()
        self.sentence_permuters = sentence_permuters
        self.sentence_analyzer = sentence_analyzer

    def process_treebank(self, treebank: SentenceList, **kwargs):
        for permuter in self.sentence_permuters:
            new_treebank = copy.deepcopy(treebank)  # Avoids modifying previous output
            for sentence in new_treebank:
                permuted_sentence = permuter.process_sentence(sentence, **kwargs)
                analyzed_sentence = self.sentence_analyzer.process_sentence(
                    permuted_sentence
                )
                yield analyzed_sentence
