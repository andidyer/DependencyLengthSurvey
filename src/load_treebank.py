from pathlib import Path
from typing import List, Dict, AnyStr
import conllu
from conllu.models import Token, TokenList, SentenceList

from src.sentence_cleaner import SentenceCleaner


class TreebankLoader:
    """ ""Loads a Treebank"""

    def __init__(
        self,
            remove_config: List[Dict] = None,
            fields_to_remove: List[AnyStr] = None,
            min_len: int = 1,
            max_len: int = 999,
            mask_words: bool = False,
    ):
        self.cleaner = SentenceCleaner(remove_config, fields_to_remove, mask_words=mask_words)
        self.min_len = min_len
        self.max_len = max_len

    def load_treebank(self, infile: Path):
        sentences = self.iter_load_treebank(infile)
        return SentenceList(sentences)

    def iter_load_treebank(self, infile: Path):
        with open(infile, encoding="utf-8") as fin:
            sentence_generator = conllu.parse_incr(fin)
            for sentence in sentence_generator:
                sentence = self.cleaner(sentence)
                if (
                    self._filter_with_sanity_checks(sentence)
                    and self.min_len <= len(sentence) <= self.max_len
                ):
                    yield sentence

    def _filter_with_sanity_checks(self, sentence: TokenList):
        checks = [
            SanityChecks.sentence_has_single_root(sentence),
            SanityChecks.sentence_has_no_orphans(sentence),
        ]
        if all(checks):
            return True

    def _filter_with_length_limits(self, sentence: TokenList):
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
