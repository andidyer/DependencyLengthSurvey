from abc import ABC, abstractmethod
from conllu import TokenList


class SentenceProcessor(ABC):

    @abstractmethod
    def process_sentence(self, sentence: TokenList, **kwargs):
        pass
