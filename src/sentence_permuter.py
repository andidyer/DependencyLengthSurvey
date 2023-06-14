import random
from typing import Callable, List, Dict
from collections import defaultdict

from conllu.models import TokenList, TokenTree

from src.utils.treeutils import get_tree_weight, Node
from src.utils.decorators import (
    preserve_metadata,
    fix_token_indices,
    deepcopy_tokenlist,
)
from src.utils.abstractclasses import SentenceMainProcessor


class SentencePermuter(SentenceMainProcessor):
    fileext = ".conllu"

    _shuffle_left = False
    _shuffle_right = False
    _reverse_left = False
    _reverse_right = False

    @deepcopy_tokenlist
    def process_sentence(self, sentence: TokenList, **kwargs):
        permuted_sentence = self.permutation_function(sentence)
        return permuted_sentence

    @preserve_metadata
    @fix_token_indices
    def permutation_function(self, sentence: TokenList):
        sentence_tree = sentence.to_tree()
        nodetree = self.build_tree(sentence_tree)
        new_sentence = self._make_new_tokenlist_from_tree(nodetree)
        return new_sentence

    def _make_new_tokenlist_from_tree(self, nodetree: Node):
        tokens = list(nodetree.traverse())
        new_sentence = TokenList(tokens)
        return new_sentence

    def build_tree(self, tokentree: TokenTree):
        left = []
        right = []

        children = self._ordering_function(tokentree.children)

        for i, subtree in enumerate(children):
            new_branch = self.build_tree(subtree)
            branch_direction: int = self._directionality_function(subtree)

            if branch_direction < 0:
                left.append(new_branch)
            elif branch_direction > 0:
                right.append(new_branch)
            else:
                raise ValueError("Directionality function must return [-1,1]")

        if self._shuffle_left:
            random.shuffle(left)
        if self._shuffle_right:
            random.shuffle(right)
        if self._reverse_left:
            left.reverse()
        if self._reverse_right:
            right.reverse()

        permutation_tree = Node.make_node(
            centre=tokentree.token, left=left, right=right
        )
        return permutation_tree

    def _directionality_function(self, subtree: TokenTree) -> int:
        # Must be overriden with a function that takes subtree as an argument and returns [-1,1]
        head_position = subtree.token["head"]
        dependent_position = subtree.token["id"]

        if dependent_position < head_position:
            return -1
        else:
            return 1

    def _ordering_function(self, tree_children: List[TokenTree]):
        # Override this as necessary
        return tree_children


class RandomProjectivePermuter(SentencePermuter):

    _shuffle_left = True
    _shuffle_right = True

    def _directionality_function(self, subtree: TokenTree, **kwargs) -> int:
        return random.choice([-1, 1])


class RandomSameSidePermuter(SentencePermuter):
    """Keeps all nodes on same side but shuffles order"""

    _shuffle_left = True
    _shuffle_right = True


class RandomSameValencyPermuter(SentencePermuter):
    """Random order but same number of nodes on each side as in observed"""

    _shuffle_left = True
    _shuffle_right = True

    @deepcopy_tokenlist
    def build_tree(self, tokentree: TokenTree):
        left = []
        right = []

        children = self._ordering_function(tokentree.children)

        # Find the number of tokens that can be on the left
        left_branches: int = sum(
            1 for child in children if child.token["id"] < child.token["head"]
        )

        for i, subtree in enumerate(children):
            new_branch = self.build_tree(subtree)
            branch_direction: int = self._directionality_function(
                subtree, i, left_branches
            )

            if branch_direction < 0:
                left.append(new_branch)
            elif branch_direction > 0:
                right.append(new_branch)
            else:
                raise ValueError("Directionality function must return [-1,1]")

        if self._shuffle_left:
            random.shuffle(left)
        if self._shuffle_right:
            random.shuffle(right)
        if self._reverse_left:
            left.reverse()
        if self._reverse_right:
            right.reverse()

        permutation_tree = Node.make_node(
            centre=tokentree.token, left=left, right=right
        )
        return permutation_tree

    def _directionality_function(
        self, subtree: TokenTree, i: int = 0, n_left: int = 0
    ) -> int:
        if i < n_left:
            return -1
        else:
            return 1

    def _ordering_function(self, tree_children: List[TokenTree]):
        return random.sample(tree_children, len(tree_children))


class OptimalProjectivePermuter(SentencePermuter):
    """Permutes according to optimal ordering (inside out by subtree weight"""

    _reverse_left = True

    def build_tree(self, tokentree: TokenTree, isRight=False):
        left = []
        right = []

        children = self._ordering_function(tokentree.children)

        nChildrenIsOdd = len(children) % 2 != 0

        for i, subtree in enumerate(children):

            # We need to reverse the direction in light of number of children
            if isRight and nChildrenIsOdd:
                i += 1
            elif not isRight and not nChildrenIsOdd:
                i += 1

            branch_direction: int = self._directionality_function(subtree, i)

            if branch_direction < 0:
                new_branch = self.build_tree(subtree, isRight=False)
                left.append(new_branch)
            elif branch_direction > 0:
                new_branch = self.build_tree(subtree, isRight=True)
                right.append(new_branch)
            else:
                raise ValueError("Directionality function must return [-1,1]")

        if self._shuffle_left:
            random.shuffle(left)
        if self._shuffle_right:
            random.shuffle(right)
        if self._reverse_left:
            left.reverse()
        if self._reverse_right:
            right.reverse()

        permutation_tree = Node.make_node(
            centre=tokentree.token, left=left, right=right
        )
        return permutation_tree

    def _directionality_function(self, subtree: TokenTree, i=0) -> int:
        if i % 2 == 0:
            return -1
        else:
            return 1

    def _ordering_function(self, tree_children: List[TokenTree]):
        return sorted(tree_children, key=get_tree_weight)


class FixedOrderPermuter(SentencePermuter):
    _reverse_left = True

    def __init__(self, grammar: Dict):
        super().__init__()
        self.grammar = defaultdict(float, grammar)

    def _lookup_deprel(self, subtree: TokenTree):
        return self.grammar[subtree.token["deprel"]]

    def _ordering_function(self, tree_children: List[TokenTree]):
        return sorted(tree_children, key=lambda child: abs(self._lookup_deprel(child)))

    def _directionality_function(self, subtree: TokenTree) -> int:
        position_value: float = self._lookup_deprel(subtree)
        if position_value < 0:
            return -1
        else:
            return 1
