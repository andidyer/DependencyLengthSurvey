import random
from typing import Callable
from collections import defaultdict
from itertools import cycle

from conllu.models import Token, TokenList, TokenTree, SentenceList

from src.utils.treeutils import fix_token_indices, get_tree_weight, Node
from src.utils.abstractclasses import SentenceMainProcessor


class SentencePermuter(SentenceMainProcessor):
    """ "Object that permutes a sentence according to a given"""

    random_floats_cycle = cycle(random.uniform(-1, 1) for i in range(100))

    def __init__(self, mode: str):
        self._initialize_dependency_positions()
        self.mode = self._set_permutation_function(mode)

    def _initialize_dependency_positions(self):
        self.dependency_positions = defaultdict(lambda: next(self.random_floats_cycle))

    def _set_permutation_function(self, mode: str):
        mode_dict = {
            "random_nonprojective": self.random_nonprojective_permute,
            "random_projective": self.random_projective_permute,
            "random_projective_fixed": self.random_projective_fixed_permute,
            "random_same_valency": self.random_same_valency_permute,
            "random_same_side": self.random_same_side_permute,
            "optimal_projective": self.optimal_projective_permute,
            "optimal_projective_weight": self.optimal_projective_weight_permute,
            "original_order": self.original_order_permute,
        }
        try:
            self.permutation_function = mode_dict[mode]
        except KeyError:
            raise KeyError(
                """Unrecognised permutation type. Choose from:\n
                                - random_nonprojective
                                - random_projective
                                - random_projective_fixed
                                - random_same_valency
                                - random_same_side
                                - optimal_projective
                                - optimal_projective_weight
                                - original_order"""
            )

    def process_sentence(self, sentence: TokenList, printable=False, **kwargs):
        permuted_sentence = self.permutation_function(sentence)
        return permuted_sentence

    def _sentence_permute_base(
        self, permutation_inner_function: Callable, sentence: TokenList, **kwargs
    ):
        sentence_tree = sentence.to_tree()
        permutation_tree = permutation_inner_function(sentence_tree, **kwargs)
        new_sentence = TokenList(
            list(permutation_tree.traverse()), metadata=sentence.metadata
        )
        new_sentence = fix_token_indices(new_sentence)
        return new_sentence

    def random_nonprojective_permute(self, sentence: TokenList):
        """permutes a sentence completely randomly with no regard for projectivity"""
        randomized_tokens = random.sample(sentence, k=len(sentence))
        new_sentence = TokenList(randomized_tokens, metadata=sentence.metadata)
        new_sentence = fix_token_indices(new_sentence)
        return new_sentence

    def random_projective_permute(self, sentence: TokenList):
        """permutes a sentence in random projective order"""
        # Make tree from sentence root
        return self._sentence_permute_base(
            self._random_projective_construct_tree, sentence
        )

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

        permutation_tree = Node.make_node(centre=tree.token, left=left, right=right)
        return permutation_tree

    def random_projective_fixed_permute(self, sentence: TokenList):
        """permutes a sentence in a projective order randomised according to dependency type"""
        return self._sentence_permute_base(
            self._random_projective_fixed_construct_tree, sentence
        )

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

        permutation_tree = Node.make_node(centre=tree.token, left=left, right=right)
        return permutation_tree

    def random_same_valency_permute(self, sentence: TokenList):
        """
        :param sentence: A tokenlist sentence
        :return: A new permuted sentence

        Permutes the sentence randomly with the constraint that any head must have the same number of subtrees on each
        side as in the original sentence.
        """
        return self._sentence_permute_base(
            self._random_same_valency_construct_tree, sentence
        )

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

        permutation_tree = Node.make_node(centre=tree.token, left=left, right=right)
        return permutation_tree

    def random_same_side_permute(self, sentence: TokenList):
        """
        :param sentence: A tokenlist sentence
        :return: A new permuted sentence

        Permutes the sentence randomly with the constraint that any node must be on the same side of
        its head as in the original sentence.
        """
        return self._sentence_permute_base(
            self._random_same_side_construct_tree, sentence
        )

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

        permutation_tree = Node.make_node(centre=tree.token, left=left, right=right)
        return permutation_tree

    def optimal_projective_permute(self, sentence: TokenList):
        return self._sentence_permute_base(
            self._optimal_projective_construct_tree, sentence, metric="distance"
        )

    def optimal_projective_weight_permute(self, sentence: TokenList):
        return self._sentence_permute_base(
            self._optimal_projective_construct_tree, sentence, metric="weight"
        )

    def _optimal_projective_construct_tree(self, tree: TokenTree, metric="distance"):
        left = []
        right = []

        head_position = tree.token["id"]

        if metric == "distance":
            sorting_key = lambda child: abs(
                subtree_head_position(child) - head_position
            )
        elif metric == "weight":
            sorting_key = get_tree_weight
        else:
            raise ValueError("Invalid distance metric")

        sorted_children = sorted(
            tree.children,
            key=sorting_key,
        )

        for i, subtree in enumerate(sorted_children):
            if i % 2 == 0:
                left.append(
                    self._optimal_projective_construct_tree(subtree, metric=metric)
                )
            else:
                right.append(
                    self._optimal_projective_construct_tree(subtree, metric=metric)
                )
        # Reverse left so that shortest are closer to head
        left = left[::-1]

        permutation_tree = Node.make_node(centre=tree.token, left=left, right=right)
        return permutation_tree

    def original_order_permute(self, sentence: TokenList):
        return self._sentence_permute_base(
            self._original_order_construct_tree, sentence
        )

    def _original_order_construct_tree(self, tree: TokenTree):
        left = []
        right = []

        head_position = tree.token["id"]

        for subtree in tree.children:
            dependent_position = subtree_head_position(subtree)
            if dependent_position < head_position:
                left.append(self._random_same_side_construct_tree(subtree))
            else:
                right.append(self._random_same_side_construct_tree(subtree))

        permutation_tree = Node.make_node(centre=tree.token, left=left, right=right)
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
            permuted_sentence = self.sentence_permuter.process_sentence(sentence)
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
