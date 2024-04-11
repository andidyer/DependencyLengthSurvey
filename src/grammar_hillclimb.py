from __future__ import annotations
import random
import copy
import logging
import math
import sys

import numpy as np
from typing import List, Callable, Generator, Union, Dict, SupportsAbs

from collections import Counter
from dataclasses import dataclass, field
from conllu import TokenList

from src.sentence_analyzer import SentenceAnalyzer
from src.sentence_permuter import FixedOrderPermuter
from src.model_treebanks import BigramMutualInformationModeller


EPSILON_HI = 1e-6
EPSILON_LO = 1e-7


class Analyzer:

    pass

class SentenceLevelAnalyzer(Analyzer):

    metric = "NULL"

    def __init__(self, **kwargs):
        self.previous_rawscore = EPSILON_HI
        self.previous_wordcount = EPSILON_LO

        self.current_rawscore = EPSILON_HI
        self.current_wordcount = EPSILON_LO

    def ingest_sentence(self, sentence: TokenList):
        pass

    def update_previous_score(self):
        self.previous_rawscore = self.current_rawscore
        self.previous_wordcount = self.current_wordcount

    def get_previous_score(self):
        return self.previous_rawscore / self.previous_wordcount

    def get_current_score(self):
        return self.current_rawscore / self.current_wordcount

    def get_improvement_score(self):
        cs, ps = self.get_current_score(), self.get_previous_score()
        imp_score = cs / ps
        logging.debug(f"{cs}/{ps}={imp_score}")
        return imp_score

    def use_previous_score(self):
        self.current_rawscore = self.previous_rawscore
        self.current_wordcount = self.previous_wordcount

    def flush(self):
        self.current_wordcount = EPSILON_LO
        self.current_rawscore = EPSILON_HI


class IntervenerComplexityAnalyzer(SentenceLevelAnalyzer):

    metric = "ICM"

    def __init__(self, **kwargs):
        super().__init__()
        self.analyzer = SentenceAnalyzer(["IntervenerComplexity"], aggregate=True)

    def analyze_sentence(self, sentence: TokenList):
        response = self.analyzer.process_sentence(sentence)
        length = response["Length"]
        metric = response["ICM"]
        return metric, length

    def ingest_sentence(self, sentence: TokenList):
        metric, length = self.analyze_sentence(sentence)
        self.current_rawscore += metric
        self.current_wordcount += length


class DLAnalyzer(SentenceLevelAnalyzer):

    metric = "DL"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analyzer = SentenceAnalyzer(["DependencyLength"], aggregate=True)

    def analyze_sentence(self, sentence: TokenList):
        response = self.analyzer.process_sentence(sentence)
        length = response["Length"]
        metric = response["DL"]
        return metric, length

    def ingest_sentence(self, sentence: TokenList):
        metric, length = self.analyze_sentence(sentence)
        self.current_rawscore += metric
        self.current_wordcount += length


class CorpusLevelAnalyzer(Analyzer):

    metric = "NULL"

    def __init__(self, **kwargs):
        self.previous_rawscore = EPSILON_LO
        self.current_rawscore = EPSILON_HI

    def ingest_sentence(self, sentence: TokenList):
        pass

    def update_previous_score(self):
        self.previous_rawscore = self.current_rawscore

    def get_previous_score(self):
        return self.previous_rawscore

    def get_current_score(self):
        return self.current_rawscore

    def get_improvement_score(self):
        cs, ps = self.get_current_score(), self.get_previous_score()
        imp_score = ps / cs
        logging.debug(f"{ps}/{cs}={imp_score}")
        return imp_score

    def use_previous_score(self):
        self.current_rawscore = self.previous_rawscore

    def flush(self):
        self.modeller.flush()
        self.current_score = EPSILON_LO


class BigramMutualInformationAnalyzer(CorpusLevelAnalyzer):

    metric = "BigramMI"

    def __init__(self, lowercase=True, threshold=2, normalized=False, **kwargs):
        super().__init__(**kwargs)
        self.modeller = BigramMutualInformationModeller(lowercase=lowercase, threshold=threshold, normalized=normalized)

    def ingest_sentence(self, sentence: TokenList):
        self.modeller._ingest_sentence(sentence)

    def get_improvement_score(self):
        cs, ps = self.get_current_score(), self.get_previous_score()
        imp_score = ps / cs
        logging.debug(f"{ps}/{cs}={imp_score}")
        return imp_score

    def get_current_score(self):
        self.calculate_current_score()
        return self.current_rawscore

    def calculate_current_score(self):
        self.current_rawscore = self.modeller.model_score()


class AnalyzerFactory:

    analyzer_dict = {
        "DependencyLength": DLAnalyzer,
        "IntervenerComplexity": IntervenerComplexityAnalyzer,
        "BigramMutualInformation": BigramMutualInformationAnalyzer,
    }

    @staticmethod
    def create_analyzer(analyzer_name: str, **kwargs):
        return AnalyzerFactory.analyzer_dict[analyzer_name](**kwargs)

    @staticmethod
    def create_analyzers(analyzer_names: List[str], **kwargs):
        return list(AnalyzerFactory.create_analyzer(analyzer_name, **kwargs) for analyzer_name in analyzer_names)


@dataclass(frozen=True)
class GrammarContainer:
    grammar: dict
    epoch: int

    def update_epoch(self, epoch: int):
        return GrammarContainer(grammar=self.grammar, epoch=epoch)


@dataclass(frozen=True)
class TrainGrammarContainer(GrammarContainer):
    train_scores: TrainScore
    dev_scores: DevScore
    epoch: int
    name: str = field(init=False, default="Train")
    update: bool = False
    inert: bool = False


@dataclass(frozen=True)
class BaselineGrammarContainer(GrammarContainer):
    train_scores: TrainScore
    dev_scores: DevScore
    name: str = field(init=False, default="Baseline")


@dataclass(frozen=True)
class Score:
    metric_scores: dict


@dataclass(frozen=True)
class TrainScore(Score):
    improvements: dict


@dataclass(frozen=True)
class DevScore(Score):
    pass


def _relative_order_same(grammar1: Dict[float], grammar2: Dict[float]):
    order1 = sorted(grammar1, key=lambda x: grammar1[x])
    order2 = sorted(grammar2, key=lambda x: grammar2[x])
    return order1 == order2


def _coerce_sentence_list_to_iterable(
    sentences: Union[List, Callable]
) -> Union[List, Generator]:
    if callable(sentences):
        return (sent for sent in sentences())
    elif isinstance(sentences, list):
        return sentences


def _change_grammar_parameters_poisson(
    grammar: dict, sample_weights: list = None, lam: float = 1.5
):

    grammar_copy = copy.deepcopy(grammar)

    if sample_weights is None:
        sample_weights = list(1 / len(grammar_copy) for _ in grammar_copy)

    n_change_params = np.random.poisson(lam)
    n_change_params = np.clip(n_change_params, 1, len(grammar_copy)).item()
    assert n_change_params >= 1, "Zero change params"

    keys = list(grammar.keys())

    params_to_change = np.random.choice(
        keys, n_change_params, replace=False, p=sample_weights
    )
    logging.debug(f"Changing params: {', '.join(params_to_change)}")

    for param in params_to_change:
        grammar_copy[param] = random.uniform(-1, 1)

    return grammar_copy


def _change_grammar_parameters_int(
    grammar: dict, n_params: int, sample_weights: list = None
):

    grammar_copy = copy.deepcopy(grammar)

    if sample_weights is None:
        sample_weights = list(1 / len(grammar_copy) for _ in grammar_copy)

    if not 1 <= n_params <= len(grammar_copy):
        raise ValueError(f"n_params {n_params} not within 1 and total size of grammar")

    keys = list(grammar.keys())

    params_to_change = np.random.choice(keys, n_params, replace=False, p=sample_weights)
    logging.debug(f"Changing params: {', '.join(params_to_change)}")

    for param in params_to_change:
        grammar_copy[param] = random.uniform(-1, 1)

    return grammar_copy


def _change_grammar_parameters(
    grammar: dict,
    n_params: int = 1,
    sample_weights: list = None,
    poisson: bool = False,
    lam: float = 1,
):
    """
    :param grammar: The grammar to change parameters of. Expects Dict[str, float].
    :param n_params: The number of parameters to change. Must be between 1 and the total number of parameters.
    :param poisson: Whether to use a poisson PNG to decide how many parameters to change. Mutex with n_params.
    :param lam: Lambda value for the poisson PNG if used.
    :return: dict
    """

    if poisson:
        return _change_grammar_parameters_poisson(
            grammar, lam=lam, sample_weights=sample_weights
        )
    else:
        return _change_grammar_parameters_int(
            grammar, n_params, sample_weights=sample_weights
        )


class GrammarHillclimb:
    def __init__(
        self,
        deprels: List[str],
        analyzers: List[Analyzer],
        objective_weights: List[SupportsAbs] = None,
    ):

        self.deprels = deprels
        self.train_analyzers = analyzers
        self.dev_analyzers = list(
            copy.deepcopy(analyzer) for analyzer in self.train_analyzers
        )
        self.baseline_analyzers = list(
            copy.deepcopy(analyzer) for analyzer in self.train_analyzers
        )
        self.analyzer_weights = objective_weights

    def _set_deprel_probability_weights(self, train_sentences: Union[List, Callable]):

        train_sentences = _coerce_sentence_list_to_iterable(train_sentences)

        # Use dict not counter because we need to constrain the entries
        raw_frequencies = {deprel: 1 for deprel in self.deprels}

        for sentence in train_sentences:
            for token in sentence:
                raw_frequencies[token["deprel"]] += 1

        # Convert to probabilities
        values_array = np.asarray(list(raw_frequencies.values()))
        sample_weights = values_array / np.sum(values_array)
        self.deprel_weights = sample_weights

    def _train_grammar_step(
        self,
        grammar: dict,
        train_sentences: Union[List, Callable],
        dev_sentences: Union[List, Callable, None],
        epoch=-1,
    ):

        # Reset grammar
        hypothesis_grammar = _change_grammar_parameters(
            grammar, poisson=True, lam=1.0, sample_weights=self.deprel_weights
        )

        # Create new fixed order permuter with new grammar
        hypothesis_permuter = FixedOrderPermuter(hypothesis_grammar)

        # Whether the grammar will get an update because of the change. Default false.
        update = False
        inert = False

        # Skip and return previous epoch if relative order is unaffected
        if _relative_order_same(grammar, hypothesis_grammar) and epoch > 0:

            logging.debug("relative order same; skipping computation")

            for analyzer in self.train_analyzers + self.dev_analyzers:
                analyzer.use_previous_score()

            inert = True

        else:

            # Load the train sentences
            train_sentences = _coerce_sentence_list_to_iterable(train_sentences)

            for sentence in train_sentences:

                # Permute sentence with fixed order permuter
                permuted_sentence = hypothesis_permuter.process_sentence(sentence)

                # Analyzers ingest sentence
                for analyzer in self.train_analyzers:
                    analyzer.ingest_sentence(permuted_sentence)

        # Get mean improvement score as weighted arithmetic mean
        improvement_scores = {
            analyzer.metric: analyzer.get_improvement_score()
            for analyzer in self.train_analyzers
        }
        mean_improvement_score = np.average(
            list(improvement_scores.values()), weights=self.analyzer_weights
        )

        # Get train metric scores
        train_metric_scores = {
            analyzer.metric: analyzer.get_current_score()
            for analyzer in self.train_analyzers
        }

        # Update previous training score in analyzers if MIP below 1
        if mean_improvement_score < 1.0:
            logging.debug(f"Mean improvement score of {mean_improvement_score}")
            update = True

        # Update previous scores if update is True
        if update:
            for analyzer in self.train_analyzers:
                analyzer.update_previous_score()

        # Flush current scores
        for analyzer in self.train_analyzers:
            analyzer.flush()

        train_scores = TrainScore(
            metric_scores=train_metric_scores, improvements=improvement_scores
        )

        # Dev evaluation
        if dev_sentences is None:
            dev_metric_scores = {
                analyzer.metric: None for analyzer in self.dev_analyzers
            }

        elif inert:
            dev_metric_scores = {
                analyzer.metric: analyzer.get_previous_score()
                for analyzer in self.dev_analyzers
            }

        else:

            if callable(dev_sentences):
                dev_sentences = (sent for sent in dev_sentences())

            for sentence in dev_sentences:

                permuted_sentence = hypothesis_permuter.process_sentence(sentence)

                for analyzer in self.dev_analyzers:
                    analyzer.ingest_sentence(permuted_sentence)

            dev_metric_scores = {
                analyzer.metric: analyzer.get_current_score()
                for analyzer in self.dev_analyzers
            }
            for analyzer in self.dev_analyzers:
                if update:
                    analyzer.update_previous_score()
                analyzer.flush()

        dev_scores = DevScore(metric_scores=dev_metric_scores)

        return TrainGrammarContainer(
            grammar=hypothesis_grammar,
            train_scores=train_scores,
            dev_scores=dev_scores,
            update=update,
            inert=inert,
            epoch=epoch,
        )

    def _baseline_evaluation(
        self,
        baseline_grammar: dict,
        train_sentences: Union[List, Callable],
        dev_sentences: [Union, Callable, None],
    ):

        # Make the hypothesis permuter
        hypothesis_permuter = FixedOrderPermuter(baseline_grammar)

        # Load the train sentences
        train_sentences = _coerce_sentence_list_to_iterable(train_sentences)

        for sentence in train_sentences:

            # Permute sentence with fixed order permuter
            permuted_sentence = hypothesis_permuter.process_sentence(sentence)

            # Analyzers ingest sentence
            for analyzer in self.train_analyzers:
                analyzer.ingest_sentence(permuted_sentence)

        train_metric_scores = {
            analyzer.metric: analyzer.get_current_score()
            for analyzer in self.train_analyzers
        }
        train_scores = TrainScore(metric_scores=train_metric_scores, improvements=None)

        dev_sentences = _coerce_sentence_list_to_iterable(dev_sentences)

        for sentence in dev_sentences:

            permuted_sentence = hypothesis_permuter.process_sentence(sentence)

            for analyzer in self.dev_analyzers:
                analyzer.ingest_sentence(permuted_sentence)

        dev_metric_scores = {
            analyzer.metric: analyzer.get_current_score()
            for analyzer in self.dev_analyzers
        }
        dev_scores = DevScore(metric_scores=dev_metric_scores)

        # Flush
        for analyzer in self.train_analyzers + self.dev_analyzers:
            analyzer.flush()

        return BaselineGrammarContainer(
            grammar=baseline_grammar,
            train_scores=train_scores,
            dev_scores=dev_scores,
            epoch=-1,
        )

    def train_grammars(
        self,
        train_sentences: Union[List, Callable],
        dev_sentences: Union[List, Callable] = None,
        baseline_grammar: dict = None,
        epochs=500,
        burnin=50,
    ):

        # Randomly initialise the grammar
        grammar = {deprel: random.uniform(-1, 1) for deprel in self.deprels}

        # Set deprel sample weights
        self._set_deprel_probability_weights(train_sentences)

        if baseline_grammar is not None:
            logging.info(f"Getting baseline scores")

            response = self._baseline_evaluation(
                baseline_grammar, train_sentences, dev_sentences
            )
            yield response

        logging.info(f"Beginning burn-in process: ({burnin} epochs)")
        for i in range(burnin):
            # Do not store or yield these
            logging.info(f"Burnin epoch {i}")
            response = self._train_grammar_step(grammar, train_sentences, [], epoch=i)

            if response.update:
                logging.debug(f"Changing original grammar to hypothesis grammar")
                grammar = response.grammar

        logging.info(f"Beginning generation: ({epochs} epochs)")
        for i in range(epochs):
            logging.info(f"Train epoch {i}")

            response = self._train_grammar_step(
                grammar, train_sentences, dev_sentences, epoch=i
            )

            if response.update:
                logging.debug(f"Changing original grammar to hypothesis grammar")
                grammar = response.grammar

            yield response
