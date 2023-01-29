from src.sentence_analyzer import SentenceAnalyzer
from src.sentence_permuter import *
from src.treebank_processor import TreebankPermuter, TreebankAnalyzer


def sentence_analyzer_factory(count_root=False):
    return SentenceAnalyzer(count_root=count_root)


def treebank_analyzer_factory(count_root=False):
    sentence_analyzer = sentence_analyzer_factory(count_root=count_root)
    return TreebankAnalyzer(sentence_analyzer)


def sentence_permuter_factory(mode: str, *grammar):
    # Grammar is unpacked here, but at most this will support one.
    if mode == "random_projective":
        return RandomProjectivePermuter()
    elif mode == "random_same_valency":
        return RandomSameValencyPermuter()
    elif mode == "random_same_side":
        return RandomSameSidePermuter()
    elif mode == "optimal_projective":
        return OptimalProjectivePermuter()
    elif mode == "original_order":
        return SentencePermuter()
    elif mode == "fixed_order":
        return FixedOrderPermuter(*grammar)
    else:
        raise ValueError(
            f"""Invalid permutation mode {mode} Choose from:
                                - random_projective
                                - random_same_valency
                                - random_same_side
                                - optimal_projective
                                - original_order
                                - fixed_order"""
        )


def treebank_permuter_factory(mode: str, *grammar):
    # Grammar is unpacked here, but at most these will support one.
    sentence_permuter = sentence_permuter_factory(mode, *grammar)
    return TreebankPermuter(sentence_permuter)
