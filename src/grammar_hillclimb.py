from __future__ import annotations
import random
import copy
import logging
import statistics
from typing import List, Callable, Iterable, Union

from dataclasses import dataclass
from conllu import TokenList

from src.sentence_analyzer import SentenceAnalyzer
from src.sentence_permuter import FixedOrderPermuter, OptimalProjectivePermuter


class Analyzer:
    def ingest_sentence(self, sentence: TokenList):
        pass

    def getScore(self):
        pass


class MemorySurprisalAnalyzer(Analyzer):
    pass


class DLMAnalyzer(Analyzer):
    @dataclass
    class ScoreContainer:
        optimal_DL: int = 0
        grammar_DL: int = 0

        def flush(self):
            self.optimal_DL: int = 0
            self.grammar_DL: int = 0

    def __init__(self, initial_grammar: dict):
        self.grammar = initial_grammar

        self.analyzer = SentenceAnalyzer(["DependencyLength"], aggregate=True)
        self.optimal_permuter = OptimalProjectivePermuter()
        self.grammar_permuter = FixedOrderPermuter(initial_grammar)

        self.scores = self.ScoreContainer()

    def ingest_sentence(self, sentence: TokenList):

        grammar_permutation = self.grammar_permuter.process_sentence(sentence)
        optimal_permutation = self.optimal_permuter.process_sentence(sentence)

        grammar_analysis = self.analyzer.process_sentence(grammar_permutation)
        optimal_analysis = self.analyzer.process_sentence(optimal_permutation)

        self.scores.optimal_DL += optimal_analysis["DL"]
        self.scores.grammar_DL += grammar_analysis["DL"]

    def getScore(self):
        try:
            score = self.scores.optimal_DL / self.scores.grammar_DL
        except ZeroDivisionError:
            score = 0
        return score

    def flush(self):
        self.scores.flush()

    def set_grammar(self, grammar):
        self.grammar = grammar
        self.grammar_permuter.grammar = grammar

    def overwrite_results(self, other: DLMAnalyzer):
        self.scores.optimal_DL = other.scores.optimal_DL
        self.scores.grammar_DL = other.scores.grammar_DL


class ProcessorCell:
    def __init__(self, analyzers: List[Analyzer], grammar: dict, ID: int = -1):
        self.analyzers = analyzers
        self.grammar = grammar
        self.id = ID

        self._relative_order_same = False

    def ingest_sentence(self, sentence: TokenList):
        # kwargs argument allows for things such as spedifying the changed deprel so that
        # analyzers that only need sentences with that deprel can skip others

        if self._relative_order_same:
            return

        for analyzer in self.analyzers:
            analyzer.ingest_sentence(sentence)

    def getScores(self):
        scores = tuple(analyzer.getScore() for analyzer in self.analyzers)
        return scores

    def getMeanScore(self):
        scores = self.getScores()
        meanscore = statistics.harmonic_mean(scores)
        return meanscore

    def flush(self):
        for analyzer in self.analyzers:
            analyzer.flush()
        self.unsetRelativeOrderSame()

    def set_grammar(self, new_grammar: dict):

        self.grammar = new_grammar
        for analyzer in self.analyzers:
            analyzer.set_grammar(new_grammar)

    def setRelativeOrderSame(self):
        self._relative_order_same = True

    def unsetRelativeOrderSame(self):
        self._relative_order_same = False

    def getRelativeOrderSame(self):
        return self._relative_order_same


@dataclass(frozen=True)
class GrammarContainer:
    grammar: dict
    score: float
    stage: str
    epoch: int = -1
    processorNo: int = -1

    def update_epoch(self, epoch: int):
        return GrammarContainer(self.grammar, self.score, self.stage, epoch=epoch)


def _change_grammar(field: str, value: float, grammar: dict):
    new_grammar = copy.deepcopy(grammar)
    new_grammar[field] = value
    return new_grammar


class GrammarHillclimb:
    def __init__(self, deprels: List[str], swarmsize: int = 1):

        self.deprels = deprels

        # Make baseline and hypothesis processor rows
        self.baseline_processors = make_processor_cells(
            ["DLMAnalyzer"], deprels, swarmsize
        )
        self.dev_processors = make_processor_cells(["DLMAnalyzer"], deprels, swarmsize)
        self.hypothesis_processors = make_processor_cells(
            ["DLMAnalyzer"], deprels, swarmsize
        )

    def _train_grammar_step(self, sentences: Union[List, Callable], epoch=-1):

        if isinstance(sentences, Callable):
            sentences = sentences()
        else:
            sentences = sentences

        # Create random field and value for new hypothesis grammars
        key = random.choice(self.deprels)
        value = random.uniform(-1, 1)

        # Reset grammars in hypothesis processors
        for i, (bp, hp) in enumerate(
            zip(self.baseline_processors, self.hypothesis_processors)
        ):
            new_grammar = _change_grammar(key, value, bp.grammar)

            if _relative_orderings_equal(new_grammar, hp.grammar):
                hp.setRelativeOrderSame()
            else:
                hp.flush()

            self.hypothesis_processors[i].set_grammar(new_grammar)

        # Hypothesis processors ingest sentences
        for sentence in sentences:
            for i, hp in enumerate(self.hypothesis_processors):
                self.hypothesis_processors[i].ingest_sentence(sentence)

        # If hypothesis grammar beats working grammar, update
        for i, (bp, dp, hp) in enumerate(
            zip(
                self.baseline_processors,
                self.dev_processors,
                self.hypothesis_processors,
            )
        ):
            bp_score = bp.getMeanScore()
            hp_score = hp.getMeanScore()

            if hp_score >= bp_score:
                self.baseline_processors[i].set_grammar(hp.grammar)
                self.dev_processors[i].set_grammar(hp.grammar)
                if hp.getRelativeOrderSame():
                    dp.setRelativeOrderSame()
                else:
                    dp.flush()

                for i, (ba, da, ha) in enumerate(
                    zip(bp.analyzers, dp.analyzers, hp.analyzers)
                ):
                    ba.overwrite_results(ha)
                    da.overwrite_results(ha)

        output_containers = list(
            GrammarContainer(bp.grammar, bp.getMeanScore(), "Train", epoch, i)
            for i, bp in enumerate(self.baseline_processors)
        )

        return output_containers

    def _dev_evaluation(self, sentences: Union[List, Callable], epoch: int = -1):

        if isinstance(sentences, Callable):
            sentences = sentences()
        else:
            sentences = sentences

        # Only dev processor ingests sentences
        for sentence in sentences:
            for i, dp in enumerate(self.dev_processors):
                self.dev_processors[i].ingest_sentence(sentence)

        # If hypothesis grammar beats working grammar, update
        output_containers = list(
            GrammarContainer(dp.grammar, dp.getMeanScore(), "Dev", epoch, i)
            for i, dp in enumerate(self.dev_processors)
        )

        return output_containers

    def train_grammars(
        self,
        train_sentences: Union[List, Callable],
        dev_sentences: Union[List, Callable] = None,
        epochs=500,
        burnin=50,
    ):

        logging.info(f"Beginning burn-in process: ({burnin} epochs)")
        for i in range(burnin):
            # Do not store or yield these
            logging.info(f"Burnin epoch {i}")
            self._train_grammar_step(train_sentences)

        logging.info(f"Beginning generation: ({epochs} epochs)")
        for i in range(epochs):
            logging.info(f"Train epoch {i}")
            # Yield training steps
            yield self._train_grammar_step(train_sentences, epoch=i)

            # Yield dev steps
            yield self._dev_evaluation(dev_sentences, epoch=i)


def _relative_orderings_equal(previous_grammar: dict, new_grammar: dict):
    sorted_previous = sorted(previous_grammar, key=lambda x: previous_grammar[x])
    sorted_new = sorted(new_grammar, key=lambda x: new_grammar[x])
    return sorted_previous == sorted_new


def make_processor_cell(modes: List[str], deprels: List[str], ID: int = -1):

    random_grammar = {deprel: random.uniform(-1, 1) for deprel in deprels}

    analyzers = []
    for mode in modes:
        if mode == "DLMAnalyzer":
            analyzer = DLMAnalyzer(random_grammar)
            analyzers.append(analyzer)

    return ProcessorCell(analyzers, random_grammar, ID)


def make_processor_cells(modes: List[str], deprels: List[str], swarmsize: int = 1):

    processor_cells = []
    for i in range(swarmsize):
        processor_cell = make_processor_cell(modes, deprels, ID=i)
        processor_cells.append(processor_cell)

    return processor_cells
