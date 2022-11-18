import random
from itertools import cycle
from conllu.models import Token, TokenList, TokenTree
import typing as T
from dataclasses import dataclass, field
from collections import defaultdict

from src.sentence_cleaner import fix_tree_indices

@dataclass
class Node:
    centre: Token
    left: T.List = field(default_factory=list)
    right: T.List = field(default_factory=list)

    def traverse(self):
        for branch in self.left:
            yield from branch.traverse()
        yield self.centre
        for branch in self.right:
            yield from branch.traverse()


class Permuter:
    random_floats_cycle = cycle(random.uniform(-1,1) for i in range(1000))

    def __init__(self):
        self.dependency_positions = defaultdict(lambda: next(self.random_floats_cycle))

    def random_nonprojective_permute(self, sentence: TokenList):
        """permutes a sentence completely randomly with no regard for projectivity"""
        randomized_tokens = random.sample(sentence, k=len(sentence))
        new_sentence = TokenList(randomized_tokens, metadata=sentence.metadata)
        new_sentence = fix_tree_indices(new_sentence)
        return new_sentence

    def random_projective_permute(self, sentence: TokenList):
        """permutes a sentence in random projective order"""
        # Make tree from sentence root
        sentence_tree = sentence.to_tree()
        permutation_tree = self._random_projective_construct_tree(sentence_tree)
        new_sentence = TokenList(list(permutation_tree.traverse()), metadata=sentence.metadata)
        new_sentence = fix_tree_indices(new_sentence)
        return new_sentence

    def _random_projective_construct_tree(self, tree: TokenTree):

        left = []
        right = []

        for subtree in tree.children:
            rand = next(self.random_floats_cycle)
            if rand < 0:
                left.append(self._random_projective_construct_tree(subtree))
            else:
                right.append(self._random_projective_construct_tree(subtree))
        random.shuffle(left)
        random.shuffle(right)

        permutation_tree = Node(centre=tree.token, left=left, right=right)
        return permutation_tree

    def random_projective_fixed_permute(self, sentence: TokenList):
        """permutes a sentence in a projective order randomised according to dependency type"""
        sentence_tree = sentence.to_tree()
        permutation_tree = self._random_projective_fixed_construct_tree(sentence_tree)
        new_sentence = TokenList(list(permutation_tree.traverse()), metadata=sentence.metadata)
        new_sentence = fix_tree_indices(new_sentence)
        return new_sentence

    def _random_projective_fixed_construct_tree(self, tree: TokenTree):

        left = []
        right = []

        ordered_subtrees = sorted(tree.children, key=lambda tree: tree.token["id"])
        for subtree in ordered_subtrees:
            position: float = self.dependency_positions[tree.token["id"]]
            if position < 0:
                left.append(self._random_projective_construct_tree(subtree))
            else:
                right.append(self._random_projective_construct_tree(subtree))

        permutation_tree = Node(centre=tree.token, left=left, right=right)
        return permutation_tree

    def optimal_projective_permute(self, sentence: TokenList):
        raise NotImplementedError




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
