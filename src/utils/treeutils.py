import logging

from conllu import Token, TokenList, TokenTree
from typing import List
from dataclasses import dataclass, field
import time


@dataclass
class Node:
    """ "
    Utility class for constructing a tree with attention to which side branches are on
    """

    centre: Token
    left: List = field(default_factory=list)
    right: List = field(default_factory=list)

    def traverse(self):
        for branch in self.left:
            yield from branch.traverse()
        yield self.centre
        for branch in self.right:
            yield from branch.traverse()

    @classmethod
    def make_node(cls, centre: Token, left: List[TokenTree], right: List[TokenTree]):
        return Node(centre, left, right)


def get_tree_weight(tree: TokenTree):
    """ "Recursive function for getting the total number of dependents in a tree"""
    weight = 0
    for branch in tree.children:
        weight += get_tree_weight(branch)
    weight += 1

    return weight


def make_index_mapping(tokenlist: TokenList) -> dict:
    index_mapping = {0: 0, None: None}
    i = 1

    for token in tokenlist:
        if isinstance(token["id"], int):
            index_mapping[token["id"]] = i
            i += 1
    return index_mapping


def standardize_deprels(sentence: TokenList):
    new_sentence = sentence.copy()

    for i, token in enumerate(new_sentence):
        # Strips anything after the colon in a deprel. If no colon, no change.
        new_sentence[i]["deprel"] = new_sentence[i]["deprel"].split(":")[0]

    return new_sentence
