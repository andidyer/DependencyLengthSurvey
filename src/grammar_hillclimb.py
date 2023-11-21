from __future__ import annotations
import random
import copy
import logging
import statistics
import numpy as np
from typing import List, Callable, Iterable, Union, Dict, SupportsAbs

from dataclasses import dataclass,field
from conllu import TokenList

from src.sentence_analyzer import SentenceAnalyzer
from src.sentence_permuter import FixedOrderPermuter


class Analyzer:

    metric = "NULL"

    def __init__(self):
        self.previous_rawscore = None
        self.previous_wordcount = None

        self.current_rawscore = 0
        self.current_wordcount = 0

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
        if self.previous_rawscore is None:
            self.previous_rawscore = self.current_rawscore
            self.previous_wordcount = self.current_wordcount
        return self.get_current_score() / self.get_previous_score()

    def flush(self):
        self.current_wordcount = 0
        self.current_rawscore = 0


class MemorySurprisalAnalyzer(Analyzer):

    metric = "MST"


class IntervenerComplexityAnalyzer(Analyzer):

    metric = "ICM"

    def __init__(self):
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


class DLAnalyzer(Analyzer):

    metric = "DL"

    def __init__(self):
        super().__init__()
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


@dataclass(frozen=True)
class GrammarContainer:
    grammar: dict
    epoch: int

    def update_epoch(self, epoch: int):
        return GrammarContainer(grammar=self.grammar, epoch=epoch)


@dataclass(frozen=True)
class TrainGrammarContainer(GrammarContainer):
    scores: dict = field(default_factory=dict)
    improvements: dict = field(default_factory=dict)
    name: str = field(init=False, default="Train")
    update: bool = False
    inert: bool = False


@dataclass(frozen=True)
class DevGrammarContainer(GrammarContainer):
    scores: dict = field(default_factory=dict)
    name: str = field(init=False, default="Dev")
    update: bool = False
    inert: bool = False


@dataclass(frozen=True)
class BaselineGrammarContainer(GrammarContainer):
    scores: dict = field(default_factory=dict)
    name: str = field(init=False, default="Dev")


def _change_grammar(key: str, value: float, grammar: dict):
    new_grammar = copy.deepcopy(grammar)
    new_grammar[key] = value
    return new_grammar

def _relative_order_same(grammar1: Dict[float], grammar2: Dict[float]):
    order1 = sorted(grammar1, key=lambda x: grammar1[x])
    order2 = sorted(grammar2, key=lambda x: grammar2[x])
    return order1 == order2


class GrammarHillclimb:
    def __init__(self, deprels: List[str], analyzers: List[Analyzer], weights: List[SupportsAbs] = None):

        self.deprels = deprels
        self.train_analyzers = analyzers
        self.dev_analyzers = list(copy.deepcopy(analyzer) for analyzer in self.train_analyzers)
        self.analyzer_weights = weights

    def _train_grammar_step(self, grammar: dict, sentences: Union[List, Callable], epoch=-1):

        # Create random field and value for new hypothesis grammars
        key = random.choice(self.deprels)
        value = random.uniform(-1, 1)

        # Reset grammar
        new_grammar = _change_grammar(key, value, grammar)

        # Skip and return previous epoch if relative order is unaffected
        if _relative_order_same(grammar, new_grammar):

            logging.debug("relative order same; skipping computation")

            improvement_scores = {analyzer.metric: 1.0 for analyzer in self.train_analyzers}
            metric_scores = {analyzer.metric: analyzer.get_previous_score() for analyzer in self.train_analyzers}

            return TrainGrammarContainer(grammar=new_grammar, update=False, inert=True, epoch=epoch, scores=metric_scores, improvements=improvement_scores)

        # Create new fixed order permuter with new grammar
        hypothesis_permuter = FixedOrderPermuter(new_grammar)

        if callable(sentences):
            sentences = (
                sent for sent in sentences()
            )


        for sentence in sentences:

            # Permute sentence with fixed order permuter
            permuted_sentence = hypothesis_permuter.process_sentence(sentence)

            # Analyzers ingest sentence
            for analyzer in self.train_analyzers:
                analyzer.ingest_sentence(permuted_sentence)

        # Get mean improvement score as arithmetic mean
        improvement_scores = {analyzer.metric: analyzer.get_improvement_score() for analyzer in self.train_analyzers}
        mean_improvement_score = np.average(list(improvement_scores.values()), weights=self.analyzer_weights)

        # Get metric scores
        metric_scores = {analyzer.metric: analyzer.get_current_score() for analyzer in self.train_analyzers}

        update = False
        # Update previous score in analyzers if MIP below 1
        if mean_improvement_score < 1.0:
            logging.debug(f"Mean improvement score of {mean_improvement_score}; updating grammar")
            update = True
            for analyzer in self.train_analyzers:
                analyzer.update_previous_score()

        # Flush current scores
        for analyzer in self.train_analyzers:
            analyzer.flush()

        return TrainGrammarContainer(grammar=new_grammar, update=update, epoch=epoch, scores=metric_scores, improvements=improvement_scores)

    def _dev_evaluation(self, grammar: dict, sentences: Union[List, Callable], epoch: int = -1):

        # Create new fixed order permuter with new grammar
        dev_permuter = FixedOrderPermuter(grammar)

        if callable(sentences):
            sentences = (
                sent for sent in sentences()
            )

        for sentence in sentences:

            # Permute sentence with fixed order permuter
            permuted_sentence = dev_permuter.process_sentence(sentence)

            # Analyzers ingest sentence
            for analyzer in self.train_analyzers:
                analyzer.ingest_sentence(permuted_sentence)


    def train_grammars(
        self,
        train_sentences: Union[List, Callable],
        dev_sentences: Union[List, Callable] = None,
        epochs=500,
        burnin=50,

    ):

        grammar = {deprel: random.uniform(-1, 1) for deprel in self.deprels}


        logging.info(f"Beginning burn-in process: ({burnin} epochs)")
        for i in range(burnin):
            # Do not store or yield these
            logging.info(f"Burnin epoch {i}")
            response = self._train_grammar_step(grammar, train_sentences, epoch=i)

            if response.update:
                grammar = response.grammar

        logging.info(f"Beginning generation: ({epochs} epochs)")
        for i in range(epochs):
            logging.info(f"Train epoch {i}")


            response = self._train_grammar_step(grammar, train_sentences, epoch=i)
            yield response

            if response.update:
                grammar = response.grammar

            # Yield dev steps
            #yield self._dev_evaluation(dev_sentences, epoch=i)
