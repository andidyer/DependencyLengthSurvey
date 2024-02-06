from conllu import Token, TokenList, TokenTree
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterable, List
import numpy as np


EPSILON = 1e-6


class Modeller(ABC):

    @abstractmethod
    def fit(self, sentences: Iterable[TokenList], **kwargs):
        pass

    @abstractmethod
    def predict(self, sentences: Iterable[TokenList], **kwargs):
        pass

    @abstractmethod
    def fit_predict(self, sentences: Iterable[TokenList], **kwargs):
        self.fit(sentences)
        return self.predict(sentences)


class HeadDirectionEntropyModeller(Modeller):

    def __init__(self):
        self.deprel2i = defaultdict(lambda: len(self.deprel2i))
        self.counts = np.full((1, 2), EPSILON)  # Counts initialised as basically empty

    def _ingest_sentence(self, sentence: TokenList, _temp_count: np.ndarray):
        sentence_tree = sentence.to_tree()
        self._ingest_token_subtree(sentence_tree, _temp_count)

    def _ingest_token_subtree(self, tokentree: TokenTree, _temp_count: np.ndarray):
        self._ingest_token(tokentree.token, _temp_count)
        for child in tokentree.children:
            self._ingest_token_subtree(child, _temp_count)

    def _ingest_token(self, token: Token, _temp_count: np.ndarray):
        deprel_i = self.deprel2i[token["deprel"]]
        if token["head"] > token["id"]:
            _temp_count[deprel_i, 1] += 1
        else:
            _temp_count[deprel_i, 0] += 1

    def _add_new_counts(self, _temp_count: np.ndarray):
        n, k = self.counts.shape
        _temp_count[:n, :k] += self.counts
        self.counts = _temp_count[:len(self.deprel2i)]

    def _set_probabilities(self):
        self.joint_probabilities = self.counts / self.counts.sum()
        self.marginal_x = self.joint_probabilities.sum(axis=1)
        self.marginal_y = self.joint_probabilities.sum(axis=0)

    def fit(self, sentences: Iterable[TokenList], **kwargs):
        _temp_count = np.full((len(self.deprel2i)+100, 2), EPSILON)
        for sentence in sentences:
            self._ingest_sentence(sentence, _temp_count=_temp_count)
        self._add_new_counts(_temp_count)
        self._set_probabilities()

    def _predict_sentence(self, sentence: TokenList):
        probability_outcome: float = 0.0
        sentence_tree = sentence.to_tree()
        probability_outcome += self._predict_token_subtree(sentence_tree)
        return probability_outcome

    def _predict_token_subtree(self, tokentree: TokenTree):
        probability_outcome: float = self._predict_token(tokentree.token)
        for child in tokentree.children:
            probability_outcome += self._predict_token_subtree(child)
        return probability_outcome

    def _predict_token(self, token: Token):
        deprel = token["deprel"]
        deprel_i = self.deprel2i[deprel]

        marginal_x_prob = self.marginal_x[deprel_i]
        joint_prob = self.joint_probabilities[deprel_i, :]
        return np.sum(joint_prob * np.log(joint_prob / marginal_x_prob)).item()

    def predict(self, sentences: Iterable[TokenList], **kwargs):
        probability_outcome: float = 0.0
        for sentence in sentences:
            probability_outcome += self._predict_sentence(sentence)
        return -probability_outcome

    def fit_predict(self, sentences: Iterable[TokenList], **kwargs):
        self.fit(sentences)
        return self.predict(sentences)

    def model_entropy(self):
        return -np.sum(self.joint_probabilities * np.log(self.joint_probabilities / self.marginal_x[:, None]))

    def model_label_entropies(self):
        label_probs = self.joint_probabilities / self.marginal_x[:, None]
        label_ents = -np.sum(label_probs * np.log2(label_probs), axis=1)
        return {label: value for label, value in zip(self.deprel2i, label_ents)}

class BigramMutualInformationModeller(Modeller):

    def __init__(self):
        pass
