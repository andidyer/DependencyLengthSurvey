from typing import List, Generator, SupportsInt

from conllu.models import Token, TokenList, SentenceList
from src.utils.abstractclasses import SentenceMainProcessor

import json


class SentenceAnalyzer(SentenceMainProcessor):
    """Sentence-level dependency length checker"""

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
            "id": sentence.metadata["sent_id"],
            "sentlen": get_sentence_length(sentence),
            "sumdeplen": get_sentence_sum_dependency_length(sentence),
            "deplens": get_sentence_dependency_lengths(
                sentence, count_root=self.count_root
            ),
            "icm": get_sentence_sum_intervener_complexity(
                sentence, count_root=self.count_root
            ),
            "interveners": get_sentence_intervener_complexities(
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


def get_sentence_sum_intervener_complexity(
    sentence: TokenList, count_root=False
) -> SupportsInt:
    complexities = get_sentence_intervener_complexities(sentence, count_root=count_root)
    return sum(complexities)


def get_sentence_intervener_complexities(
    sentence: TokenList, count_root=False
) -> List[SupportsInt]:
    """Get a list of intervener complexities for each token in the sentnece"""
    complexities = []
    heads_map = TokenList.head_to_token(sentence)
    for token in sentence:
        if not count_root and token["head"] == 0:
            continue
        dep_id = token["id"]
        head_id = token["head"]

        if dep_id < head_id:
            lower = dep_id + 1  # Must not include self
            upper = head_id + 1
        else:
            lower = head_id
            upper = dep_id

        intervening_heads = sum(
            1 for head_id in heads_map if head_id in range(lower, upper)
        )
        complexities.append(intervening_heads)

    return complexities


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
