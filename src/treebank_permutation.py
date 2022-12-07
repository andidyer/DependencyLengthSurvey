import random
import typing as T
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import cycle

from conllu.models import Token, TokenList, TokenTree, SentenceList

from src.sentence_cleaner import fix_tree_indices


@dataclass
class Node:
    """ "
    Utility class for constructing a tree with attention to which side branches are on
    """

    centre: Token
    left: T.List = field(default_factory=list)
    right: T.List = field(default_factory=list)

    def traverse(self):
        for branch in self.left:
            yield from branch.traverse()
        yield self.centre
        for branch in self.right:
            yield from branch.traverse()


class SentencePermuter:
    """ "Object that permutes a sentence according to a given"""

    random_floats_cycle = cycle(random.uniform(-1, 1) for i in range(100))

    def __init__(self, mode: str):
        self._initialize_dependency_positions()
        self.mode = self._set_permutation_function(mode)

    def _initialize_dependency_positions(self):
        self.dependency_positions = defaultdict(lambda: next(self.random_floats_cycle))

    def _set_permutation_function(self, mode: str):
        if mode == "random_nonprojective":
            self.permutation_function = self.random_nonprojective_permute
        elif mode == "random_projective":
            self.permutation_function = self.random_projective_permute
        elif mode == "random_projective_fixed":
            self.permutation_function = self.random_projective_fixed_permute
        elif mode == "random_same_valency":
            self.permutation_function = self.random_same_valency_permute
        elif mode == "random_same_side":
            self.permutation_function = self.random_same_side_permute
        elif mode == "optimal_projective":
            self.permutation_function = self.optimal_projective_permute
        else:
            raise ValueError(
                """Unrecognised permutation type. Choose from:\n
                                - random_nonprojective\n
                                - random_projective\n
                                - random_projective_fixed\n
                                - random_same_valency\n
                                - optimal_projective"""
            )

    def permute_sentence(self, sentence: TokenList):
        return self.permutation_function(sentence)

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
        new_sentence = TokenList(
            list(permutation_tree.traverse()), metadata=sentence.metadata
        )
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
        new_sentence = TokenList(
            list(permutation_tree.traverse()), metadata=sentence.metadata
        )
        new_sentence = fix_tree_indices(new_sentence)
        return new_sentence

    def _random_projective_fixed_construct_tree(self, tree: TokenTree):

        left = []
        right = []

        ordered_subtrees = sorted(
            tree.children, key=lambda subtree: subtree_head_position(subtree)
        )
        for subtree in ordered_subtrees:
            position: float = self.dependency_positions[subtree.token["deprel"]]
            if position < 0:
                left.append(self._random_projective_fixed_construct_tree(subtree))
            else:
                right.append(self._random_projective_fixed_construct_tree(subtree))

        permutation_tree = Node(centre=tree.token, left=left, right=right)
        return permutation_tree

    def random_same_valency_permute(self, sentence: TokenList):
        """
        :param sentence: A tokenlist sentence
        :return: A new permuted sentence

        Permutes the sentence randomly with the constraint that any head must have the same number of subtrees on each
        side as in the original sentence.
        """
        sentence_tree = sentence.to_tree()
        permutation_tree = self._random_same_valency_construct_tree(sentence_tree)
        new_sentence = TokenList(
            list(permutation_tree.traverse()), metadata=sentence.metadata
        )
        new_sentence = fix_tree_indices(new_sentence)
        return new_sentence

    def _random_same_valency_construct_tree(self, tree: TokenTree):

        left = []
        right = []

        head_position = tree.token["id"]
        shuffled_children = random.sample(tree.children, len(tree.children))

        # Find number of trees on left
        n_left = sum(
            1
            for subtree in shuffled_children
            if subtree_head_position(subtree) < head_position
        )

        for i, subtree in enumerate(shuffled_children):
            if i < n_left:
                left.append(self._random_same_valency_construct_tree(subtree))
            else:
                right.append(self._random_same_valency_construct_tree(subtree))
        random.shuffle(left)
        random.shuffle(right)

        permutation_tree = Node(centre=tree.token, left=left, right=right)
        return permutation_tree

    def random_same_side_permute(self, sentence: TokenList):
        """
        :param sentence: A tokenlist sentence
        :return: A new permuted sentence

        Permutes the sentence randomly with the constraint that any node must be on the same side of
        its head as in the original sentence.
        """
        sentence_tree = sentence.to_tree()
        permutation_tree = self._random_same_side_construct_tree(sentence_tree)
        new_sentence = TokenList(
            list(permutation_tree.traverse()), metadata=sentence.metadata
        )
        new_sentence = fix_tree_indices(new_sentence)
        return new_sentence

    def _random_same_side_construct_tree(self, tree: TokenTree):

        left = []
        right = []

        head_position = tree.token["id"]

        for subtree in tree.children:
            dependent_position = subtree_head_position(subtree)
            if dependent_position < head_position:
                left.append(self._random_same_side_construct_tree(subtree))
            else:
                right.append(self._random_same_side_construct_tree(subtree))
        random.shuffle(left)
        random.shuffle(right)

        permutation_tree = Node(centre=tree.token, left=left, right=right)
        return permutation_tree

    def optimal_projective_permute(self, sentence: TokenList):
        sentence_tree = sentence.to_tree()
        permutation_tree = self._optimal_projective_construct_tree(sentence_tree)
        new_sentence = TokenList(
            list(permutation_tree.traverse()), metadata=sentence.metadata
        )
        new_sentence = fix_tree_indices(new_sentence)
        return new_sentence

    def _optimal_projective_construct_tree(self, tree: TokenTree):
        left = []
        right = []

        head_position = tree.token["id"]

        sorted_children = sorted(
            tree.children,
            key=lambda child: abs(subtree_head_position(child) - head_position),
        )

        for i, subtree in enumerate(sorted_children):
            if i % 2 == 0:
                left.append(self._optimal_projective_construct_tree(subtree))
            else:
                right.append(self._optimal_projective_construct_tree(subtree))
        random.shuffle(left)
        random.shuffle(right)

        permutation_tree = Node(centre=tree.token, left=left, right=right)
        return permutation_tree


class TreebankPermuter:
    def __init__(self, mode: str, prefix: str = ""):
        self.sentence_permuter = self._get_sentence_permuter(mode)
        self.prefix = prefix
        self.mode = mode

    @staticmethod
    def _get_sentence_permuter(mode):
        return SentencePermuter(mode)

    def yield_permute_treebank(self, treebank: SentenceList):
        for sentence in treebank:
            permuted_sentence = self.sentence_permuter.permute_sentence(sentence)
            permuted_sentence.metadata["sent_id"] = (
                self.prefix + permuted_sentence.metadata["sent_id"]
            )
            yield permuted_sentence


def _is_enhanced_dependency(token: Token) -> bool:
    return (
        isinstance(token["id"], tuple)
        and token["id"][1] == "."
        and isinstance(token["id"][2], int)
    )


def _is_multiword_token(token: Token) -> bool:
    return (
        isinstance(token["id"], tuple)
        and token["id"][1] == "-"
        and isinstance(token["id"][2], int)
    )


def subtree_head_position(subtree: TokenTree):
    return subtree.token["id"]
