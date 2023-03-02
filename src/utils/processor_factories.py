from src.sentence_analyzer import SentenceAnalyzer
from src.sentence_permuter import *
from src.treebank_processor import TreebankPermuter, TreebankAnalyzer


def sentence_analyzer_factory(
    count_root=False, count_direction=False, tokenwise_scores=False
):
    return SentenceAnalyzer(
        count_root=count_root,
        count_direction=count_direction,
        tokenwise_scores=tokenwise_scores,
    )


def treebank_analyzer_factory(
    count_root=False, count_direction=False, tokenwise_scores=False
):
    sentence_analyzer = sentence_analyzer_factory(
        count_root=count_root,
        count_direction=count_direction,
        tokenwise_scores=tokenwise_scores,
    )
    return TreebankAnalyzer(sentence_analyzer)


def sentence_permuter_factory(mode: str, grammar: Dict=None):
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
        return FixedOrderPermuter(grammar)
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

def treebank_permuter_factory(mode: str, grammars: List[Dict] = None, n_times=1):
    if isinstance(grammars, list) and mode == "fixed_order":
        sentence_permuters = list(sentence_permuter_factory(mode, grammar=grammar) for grammar in grammars)
    elif mode.startswith("random"):
        sentence_permuters = list(sentence_permuter_factory(mode) for i in range(n_times))
    else:
        sentence_permuters = [sentence_permuter_factory(mode)]
    return TreebankPermuter(sentence_permuters)
