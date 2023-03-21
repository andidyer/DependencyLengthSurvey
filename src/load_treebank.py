from pathlib import Path
from typing import List, Dict, AnyStr
import conllu
from conllu.models import Token, TokenList, SentenceList

from src.sentence_cleaner import SentenceCleaner


class TreebankLoader:
    """ ""Loads a Treebank"""

    def __init__(
        self,
        cleaner: SentenceCleaner = None,
        min_len: int = 1,
        max_len: int = 999,
    ):
        if cleaner is None:
            self.cleaner = SentenceCleaner()
        else:
            self.cleaner = cleaner

        self.min_len = min_len
        self.max_len = max_len

    def load_treebank(self, infile: Path):
        sentences = self.iter_load_treebank(infile)
        return SentenceList(sentences)

    def clean_sentence(self, tokenlist: TokenList):
        return self.cleaner.process_sentence(tokenlist)

    def iter_load_treebank(self, infile: Path):
        with open(infile, encoding="utf-8") as fin:
            sentence_generator = conllu.parse_incr(fin)
            for sentence in sentence_generator:
                sentence = self.clean_sentence(sentence)
                if self.filter_with_length_limits(sentence):
                    yield sentence

    def filter_with_length_limits(self, sentence: TokenList):
        if self.min_len <= len(sentence) <= self.max_len:
            return True
        else:
            return False


class SanityChecks:
    """
    General sanity checks to make sure a oonllu sentence is not malformed
    """

    @staticmethod
    def sentence_has_single_root(sentence: TokenList):
        roots = list(filter(SanityChecks._token_is_root, sentence))
        return len(roots) == 1

    @staticmethod
    def sentence_has_no_orphans(sentence: TokenList):
        orphans = list(filter(SanityChecks._token_is_orphan, sentence))
        return len(orphans) == 0

    @staticmethod
    def _token_is_root(token: Token):
        return token["head"] == 0 and token["deprel"] == "root"

    @staticmethod
    def _token_is_orphan(token: Token):
        return token["head"] is None
