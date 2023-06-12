from conllu.models import Token, TokenList, TokenTree, SentenceList
from src.utils.abstractclasses import SentenceMainProcessor


class NullProcessor(SentenceMainProcessor):
    def process_sentence(self, sentence: TokenList, **kwargs):
        return sentence

    def process_treebank(self, treebank: SentenceList, **kwargs):
        for sentence in treebank:
            yield sentence
