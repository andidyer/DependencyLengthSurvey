from conllu import TokenList
from src.utils.abstractclasses import SentenceMainProcessor


class NullProcessor(SentenceMainProcessor):
    def process_sentence(self, sentence: TokenList, **kwargs):
        return sentence

    def process_treebank(self, treebank: list, **kwargs):
        for sentence in treebank:
            yield sentence
