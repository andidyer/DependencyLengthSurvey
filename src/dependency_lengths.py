from conllu.models import Token, TokenList, SentenceList
from typing import List, Generator


class DependencyLengthChecker:
    """Sentence-level"""

    def __init__(self, ignore_root: bool = False):
        self.ignore_root = ignore_root

    @staticmethod
    def get_pairwise_dependency_length(token: Token) -> int:
        token_position = token["id"]
        head_position = token["head"]
        dependency_length = abs(token_position - head_position)
        return dependency_length

    @staticmethod
    def get_sentence_length(sentence: TokenList) -> int:
        n_tokens = 0
        for token in sentence:
            if not isinstance(token["id"], int):
                continue
            n_tokens += 1

        return n_tokens

    def get_sentence_dependency_lengths(self, sentence: TokenList) -> List[int]:
        """Get a list of dependency lengths in the sentence"""
        lengths = self._yield_pairwise_dependency_lengths(sentence)
        return list(lengths)

    def get_sentence_sum_dependency_length(self, sentence: TokenList) -> int:
        """Get a list of dependency lengths in the sentence"""
        lengths = self._yield_pairwise_dependency_lengths(sentence)
        return sum(lengths)

    def _yield_pairwise_dependency_lengths(self, sentence: TokenList) -> Generator:
        for token in sentence:
            if self.ignore_root and token["head"] == 0:
                continue
            elif not isinstance(token["id"], int):
                continue
            elif not isinstance(token["head"], int):
                continue
            else:
                yield self.get_pairwise_dependency_length(token)


class TreebankDependencyLengthChecker(DependencyLengthChecker):
    def yield_treebank_dependency_lengths(self, treebank: SentenceList) -> Generator:
        """Yields lists of dependency lengths for all sentences in the treebank"""
        for sentence in treebank:
            yield self.get_sentence_dependency_lengths(sentence)

    def yield_treebank_sum_dependency_lengths(
        self, treebank: SentenceList
    ) -> Generator:
        for sentence in treebank:
            yield self.get_sentence_sum_dependency_length(sentence)

    def yield_treebank_sentence_lengths(self, treebank: SentenceList) -> Generator:
        for sentence in treebank:
            yield self.get_sentence_length(sentence)

    def yield_treebank_sentence_data(self, treebank: SentenceList) -> Generator:
        """yields json objects of sentence data"""
        for sentence in treebank:
            yield {
                "sentence_id": sentence.metadata["sent_id"],
                "sentence_length": self.get_sentence_length(sentence),
                "sentence_sum_dependency_length": self.get_sentence_sum_dependency_length(
                    sentence
                ),
                "sentence_dependency_lengths": self.get_sentence_dependency_lengths(
                    sentence
                ),
            }
