from abc import ABC, abstractmethod
from conllu import TokenList, SentenceList
from typing import Union, List, Iterable


class SentenceProcessor(ABC):
    @abstractmethod
    def process_sentence(self, sentence: TokenList, **kwargs):
        pass


class SentencePreProcessor(SentenceProcessor):
    @abstractmethod
    def process_sentence(self, sentence: TokenList, **kwargs):
        pass


class SentenceMainProcessor(SentenceProcessor):
    @abstractmethod
    def process_sentence(self, sentence: TokenList, **kwargs):
        pass
