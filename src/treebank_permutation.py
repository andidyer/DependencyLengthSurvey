import random
from conllu.models import Token, TokenList, TokenTree
import typing as T
from dataclasses import dataclass


@dataclass
class Node:
    centre: int
    left: T.List[T.Any]
    right: T.List[T.Any]

    def traverse(self):
        for branch in self.left:
            yield traverse(branch)
        yield self.centre
        for branch in self.right:
            yield traverse(branch)


class Permuter:
    def random_projective_permute(self, sentence: TokenList):
        pass


def linearize_token_ids(sentence: TokenList):
    """Turns token IDs into linear order, such that all token IDs are unique integers that fall
    in the sequence 1, 2, ..., L, where L is sentence length.
    Enhanced dependencies will follow their respective token, e.g. 1, 2, 2.1, 2.2, ..., 2.n, ...,L.
    Multiword tokens will begin at their earliest token, e.g. 1, 2-4, 2, 3, 4, ..., L."""
    mapping = {}
    for i, token in enumerate(sentence):
        orig_id = _get_token_id(token)
        mapping[orig_id] = i + 1

    new_sentence = reorder_tokens(mapping, sentence)


def reorder_tokens(mapping: dict, sentence: TokenList):
    """Uses a mapping in the form of a dictionary to map token IDs in an original sentence to their new positions"""
    new_sentence = sentence.copy()

    for i, token in enumerate(new_sentence):

        orig_id = new_sentence[i]["id"]
        orig_head = new_sentence[i]["head"]
        new_sentence[i]["id"] = mapping[orig_id]
        new_sentence[i]["head"] = mapping[orig_head]

    new_sentence = TokenList(sorted(new_sentence, key=_get_token_id))
    return new_sentence


def _get_token_id(token: Token):
    """Gets token id, including for multiword tokens and enhanced dependencies
    For use in sorting tokens by id"""
    if isinstance(token["id"], int):
        return token["id"]
    elif _is_enhanced_dependency(token):
        return float("".join(token["id"]))
    elif _is_multiword_token(token):
        return token["id"][0]


def _is_enhanced_dependency(token: Token):
    return (
        isinstance(token["id"], tuple)
        and token["id"][1] == "."
        and isinstance(token["id"][2], int)
    )


def _is_multiword_token(token: Token):
    return (
        isinstance(token["id"], tuple)
        and token["id"][1] == "-"
        and isinstance(token["id"][2], int)
    )
