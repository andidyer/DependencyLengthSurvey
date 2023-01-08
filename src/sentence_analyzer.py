from typing import List, Generator

from conllu.models import Token, TokenList, SentenceList
from src.utils.abstractclasses import SentenceMainProcessor

import json


class SentenceAnalyzer(SentenceMainProcessor):
    """Sentence-level dependency length checker"""

    fileext = ".ndjson"

    def __init__(self, count_root: bool = False):
        self.count_root = count_root

    def process_sentence(self, sentence: TokenList, printable=False, **kwargs):
        """ "
        Processes a  tokenlist sentence to get a json object containing the fields
            - sent_id
            - sentence_length
            - sentence_sum_dependency_length
            - sentence_dependency_lengths
        """
        sentence_data: dict = {
            "sentence_id": sentence.metadata["sent_id"],
            "sentence_length": get_sentence_length(sentence),
            "sentence_sum_dependency_length": get_sentence_sum_dependency_length(
                sentence
            ),
            "sentence_dependency_lengths": get_sentence_dependency_lengths(
                sentence, count_root=self.count_root
            ),
        }

        return sentence_data


def get_pairwise_dependency_length(token: Token) -> int:
    token_position = token["id"]
    head_position = token["head"]
    dependency_length = abs(token_position - head_position)
    return dependency_length


def get_sentence_length(sentence: TokenList) -> int:
    n_tokens = 0
    for token in sentence:
        if not isinstance(token["id"], int):
            continue
        n_tokens += 1

    return n_tokens


def get_sentence_dependency_lengths(sentence: TokenList, count_root=False) -> List[int]:
    """Get a list of dependency lengths in the sentence"""
    lengths = _yield_pairwise_dependency_lengths(sentence, count_root=count_root)
    return list(lengths)


def get_sentence_sum_dependency_length(sentence: TokenList) -> int:
    """Get a list of dependency lengths in the sentence"""
    lengths = _yield_pairwise_dependency_lengths(sentence)
    return sum(lengths)


def _yield_pairwise_dependency_lengths(
    sentence: TokenList, count_root=False
) -> Generator:
    for token in sentence:
        if not count_root and token["head"] == 0:
            continue
        elif not isinstance(token["id"], int):
            continue
        elif not isinstance(token["head"], int):
            continue
        else:
            yield get_pairwise_dependency_length(token)
