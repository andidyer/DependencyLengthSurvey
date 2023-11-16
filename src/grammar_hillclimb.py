from __future__ import annotations
import random
import copy
import logging
import statistics
from typing import List, Callable, Iterable, Union, Dict

from dataclasses import dataclass,field
from conllu import TokenList

from src.sentence_analyzer import SentenceAnalyzer
from src.sentence_permuter import FixedOrderPermuter, OptimalProjectivePermuter


class Analyzer:

    metric = "NULL"

    def __init__(self):
        self.previous_score = None
        self.current_score = 0

    def ingest_sentence(self, sentence: TokenList):
        pass

    def update_previous_score(self):
        self.previous_score = self.current_score

    def get_improvement_score(self):
        if self.previous_score is None:
            self.previous_score = self.current_score
        return self.current_score / self.previous_score

    def flush(self):
        self.current_score = 0


class MemorySurprisalAnalyzer(Analyzer):

    metric = "MST"


class IntervenerComplexityAnalyzer(Analyzer):

    metric = "ICM"

    def __init__(self):
        super().__init__()
        self.analyzer = SentenceAnalyzer(["IntervenerComplexity"], aggregate=True)

    def analyze_sentence(self, sentence: TokenList):
        return self.analyzer.process_sentence(sentence)["ICM"]

    def ingest_sentence(self, sentence: TokenList):
        self.current_score += self.analyze_sentence(sentence)


class DLAnalyzer(Analyzer):

    metric = "DL"

    def __init__(self):
        super().__init__()
        self.analyzer = SentenceAnalyzer(["DependencyLength"], aggregate=True)

    def analyze_sentence(self, sentence: TokenList):
        return self.analyzer.process_sentence(sentence)["DL"]

    def ingest_sentence(self, sentence: TokenList):
        self.current_score += self.analyze_sentence(sentence)


@dataclass(frozen=True)
class GrammarContainer:
    grammar: dict
    epoch: int

    def update_epoch(self, epoch: int):
        return GrammarContainer(grammar=self.grammar, epoch=epoch)


@dataclass(frozen=True)
class TrainGrammarContainer(GrammarContainer):
    update: bool
    scores: dict = field(default_factory=dict)
    improvements: dict = field(default_factory=dict)
    name: str = field(init=False, default="Train")


@dataclass(frozen=True)
class DevGrammarContainer(GrammarContainer):
    update: bool
    scores: dict = field(default_factory=dict)
    name: str = field(init=False, default="Dev")


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
    def __init__(self, deprels: List[str], analyzers: List[Analyzer]):

        self.deprels = deprels

        self.analyzers = analyzers

    def _train_grammar_step(self, grammar: dict, sentences: Union[List, Callable], epoch=-1):

        # Create random field and value for new hypothesis grammars
        key = random.choice(self.deprels)
        value = random.uniform(-1, 1)

        # Reset grammar
        new_grammar = _change_grammar(key, value, grammar)

        # Skip and return previous epoch if relative order is unaffected
        if _relative_order_same(grammar, new_grammar):

            logging.debug("relative order same; skipping computation")

            improvement_scores = {analyzer.metric: 1.0 for analyzer in self.analyzers}
            metric_scores = {analyzer.metric: analyzer.previous_score for analyzer in self.analyzers}

            return TrainGrammarContainer(grammar=new_grammar, update=False, epoch=epoch, scores=metric_scores, improvements=improvement_scores)

        # Create new fixed order permuter with new grammar
        hypothesis_permuter = FixedOrderPermuter(new_grammar)

        if callable(sentences):
            sentences = (sent for sent in sentences())


        for sentence in sentences:

            # Permute sentence with fixed order permuter
            permuted_sentence = hypothesis_permuter.process_sentence(sentence)

            # Analyzers ingest sentence
            for analyzer in self.analyzers:
                analyzer.ingest_sentence(permuted_sentence)

        # Get mean improvement score as arithmetic mean
        improvement_scores = {analyzer.metric: analyzer.get_improvement_score() for analyzer in self.analyzers}
        mean_improvement_score = statistics.mean(improvement_scores.values())

        # Get metric scores
        metric_scores = {analyzer.metric: analyzer.current_score for analyzer in self.analyzers}

        update = False
        # Update previous score in analyzers if MIP below 1
        if mean_improvement_score < 1.0:
            logging.debug(f"Mean improvement score of {mean_improvement_score}; updating grammar")
            update = True
            for analyzer in self.analyzers:
                analyzer.update_previous_score()

        # Flush current scores
        for analyzer in self.analyzers:
            analyzer.flush()

        return TrainGrammarContainer(grammar=new_grammar, update=update, epoch=epoch, scores=metric_scores, improvements=improvement_scores)

    def _dev_evaluation(self, sentences: Union[List, Callable], epoch: int = -1):

        pass

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

            response = self._train_grammar_step(grammar, train_sentences)
            yield response

            if response.update:
                grammar = response.grammar

            # Yield dev steps
            #yield self._dev_evaluation(dev_sentences, epoch=i)
