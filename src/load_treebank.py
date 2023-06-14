from pathlib import Path
from typing import List, Dict, AnyStr
from conllu import parse_incr, Token, TokenList, SentenceList

from src.sentence_cleaner import SentenceCleaner
from src.sentence_selector import SentenceSelector

from src.utils.decorators import (
    fix_token_indices,
    preserve_metadata,
    deepcopy_tokenlist,
)


class TreebankLoader:
    """ ""Loads a Treebank"""

    def __init__(
        self,
        cleaner: SentenceCleaner = None,
        selector: SentenceSelector = None,
        min_len: int = 1,
        max_len: int = 999,
    ):

        if cleaner is None:
            self.cleaner = SentenceCleaner()
        else:
            self.cleaner = cleaner

        if selector is None:
            self.selector = SentenceSelector()
        else:
            self.selector = selector

        self.min_len = min_len
        self.max_len = max_len

    def load_treebank(self, infile: Path):
        sentences = self.iter_load_treebank(infile)
        return SentenceList(sentences)

    def clean_sentence(self, tokenlist: TokenList):
        return self.cleaner.process_sentence(tokenlist)

    def select_tokens(self, tokenlist: TokenList):
        return self.selector.process_sentence(tokenlist)

    @deepcopy_tokenlist
    @preserve_metadata
    @fix_token_indices
    def process_sentence(self, tokenlist: TokenList):
        processed = tokenlist
        processed = self.clean_sentence(processed)
        processed = self.select_tokens(processed)
        return processed

    def iter_load_treebank(self, infile: Path):
        with open(infile, encoding="utf-8") as fin:
            sentence_generator = parse_incr(fin)
            for sentence in sentence_generator:
                sentence = self.process_sentence(sentence)

                if self.filter_with_length_limits(sentence):
                    yield sentence

    def iter_load_glob(self, indir: Path, glob_pattern: str):
        indir_path = Path(indir)
        infiles = indir_path.glob(glob_pattern)

        for infile in infiles:
            yield from self.iter_load_treebank(infile)

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
