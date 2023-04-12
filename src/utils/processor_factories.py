from src.sentence_analyzer import SentenceAnalyzer
from src.sentence_permuter import *
from src.treebank_processor import (
    TreebankPermuter,
    TreebankAnalyzer,
    TreebankPermuterAnalyzer,
)


def sentence_analyzer_factory(
    token_analyzers: List[str],
    count_root: bool = False,
    w2v: dict = None,
    language: str = None,
    aggregate: bool = False,
):
    return SentenceAnalyzer(
        token_analyzers,
        count_root=count_root,
        w2v=w2v,
        language=language,
        aggregate=aggregate,
    )


def treebank_analyzer_factory(
    token_analyzers: List[str],
    count_root: bool = False,
    w2v: dict = None,
    language: str = None,
    aggregate: bool = False,
):
    sentence_analyzer = sentence_analyzer_factory(
        token_analyzers,
        count_root=count_root,
        w2v=w2v,
        language=language,
        aggregate=aggregate,
    )
    return TreebankAnalyzer(sentence_analyzer)


def sentence_permuter_factory(mode: str, grammar: Dict = None):
    if mode == "RandomProjective":
        return RandomProjectivePermuter()
    elif mode == "RandomSameValency":
        return RandomSameValencyPermuter()
    elif mode == "RandomSameSide":
        return RandomSameSidePermuter()
    elif mode == "OptimalProjective":
        return OptimalProjectivePermuter()
    elif mode == "OriginalOrder":
        return SentencePermuter()
    elif mode == "FixedOrder":
        return FixedOrderPermuter(grammar)
    else:
        raise ValueError(
            f"""Invalid permutation mode {mode} Choose from:
                                - RandomProjective
                                - RandomSameValency
                                - RandomSameSide
                                - OptimalProjective
                                - OriginalOrder
                                - FixedOrder"""
        )


def treebank_permuter_factory(mode: str, grammars: List[Dict] = None, n_times=1):
    if isinstance(grammars, list) and mode == "FixedOrder":
        sentence_permuters = list(
            sentence_permuter_factory(mode, grammar=grammar) for grammar in grammars
        )
    elif mode.startswith("Random"):
        sentence_permuters = list(
            sentence_permuter_factory(mode) for _ in range(n_times)
        )
    else:
        sentence_permuters = [sentence_permuter_factory(mode)]
    return TreebankPermuter(sentence_permuters)


def treebank_permuter_analyzer_factory(
    permutation_mode: str,
    token_analyzers: List,
    grammars: List[Dict] = None,
    n_times=1,
    count_root: bool =False,
    w2v: dict = None,
    language: str = None,
    aggregate: bool = False,
):
    if isinstance(grammars, list) and permutation_mode == "FixedOrder":
        sentence_permuters = list(
            sentence_permuter_factory(permutation_mode, grammar=grammar) for grammar in grammars
        )
    elif permutation_mode.startswith("Random"):
        sentence_permuters = list(
            sentence_permuter_factory(permutation_mode) for i in range(n_times)
        )
    else:
        sentence_permuters = [sentence_permuter_factory(permutation_mode)]

    sentence_analyzer = sentence_analyzer_factory(
        token_analyzers,
        count_root=count_root,
        w2v=w2v,
        language=language,
        aggregate=aggregate
    )

    return TreebankPermuterAnalyzer(sentence_permuters, sentence_analyzer)
