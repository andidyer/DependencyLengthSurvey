import sys
from conllu import Token, TokenList, TokenTree
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterable, List, Tuple
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, dok_matrix


EPSILON = 1e-6
EPSILON_ARRAY = np.asarray(EPSILON)


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

    @abstractmethod
    def flush(self):
        pass


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
        self.marginal_x = self.joint_probabilities.sum(axis=0)
        self.marginal_y = self.joint_probabilities.sum(axis=1)

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

    def model_score(self):
        return -np.sum(self.joint_probabilities * np.log(self.joint_probabilities / self.marginal_x[:, None]))

    def model_label_scores(self):
        label_probs = self.joint_probabilities / self.marginal_x[:, None]
        label_ents = -np.sum(label_probs * np.log2(label_probs), axis=1)
        return {label: value for label, value in zip(self.deprel2i, label_ents)}

    def flush(self):
        self.deprel2i = defaultdict(lambda: len(self.deprel2i))
        self.counts = np.full((1, 2), EPSILON)  # Counts initialised as basically empty

class BigramMutualInformationModeller(Modeller):

    def __init__(self, lowercase: bool = True, threshold: int = 1, normalized: bool = False):
        self.w2i = defaultdict(lambda: len(self.w2i), {"<SOS>": 0, "<EOS>": 1})
        self.i2w = defaultdict(lambda: len(self.i2w), {v: k for k,v in self.w2i.items()})
        self.counts = dok_matrix((sys.maxsize, sys.maxsize)) # Initialise as empty
        self.lowercase = lowercase
        self.threshold = threshold
        self.normalized = normalized

    def _ingest_sentence(self, sentence: TokenList):
        seq = [Token(form="<SOS>")] + sentence + [Token(form="<EOS>")]
        for i, token in enumerate(seq[:-1]):
            self._ingest_bigram(seq[i], seq[i+1])

    def _ingest_bigram(self, token1: Token, token2: Token):
        form1 = self._get_form(token1, self.lowercase)
        form2 = self._get_form(token2, self.lowercase)
        w_i1 = self.w2i[form1]
        w_i2 = self.w2i[form2]
        self.i2w[w_i1] = form1
        self.i2w[w_i2] = form2
        self.counts[w_i1, w_i2] += 1

    @staticmethod
    def _get_form(token, lowercase=False):
        form = token["form"]
        return form.lower() if lowercase else form

    def _get_truncated_count_matrix(self, count_matrix: dok_matrix, max_rows: int, max_cols: int):
        new_matrix = count_matrix[:max_rows, :max_cols]
        return new_matrix

    def _thresh_counts(self, count_matrix: dok_matrix, threshold: int = 1) -> dok_matrix:

        threshed_data = {(i, j): value for (i, j), value in count_matrix.items() if value >= threshold}
        threshed_matrix = dok_matrix(count_matrix.shape)
        for (i, j), value in threshed_data.items():
            threshed_matrix[i, j] = value

        return threshed_matrix

    def _get_probabilities(self) -> Tuple[coo_matrix, np.matrix, np.matrix]:
        v = len(self.w2i)
        truncated_counts = self._get_truncated_count_matrix(self.counts, v, v)
        threshed_counts = self._thresh_counts(truncated_counts, self.threshold)
        joint_probabilities = threshed_counts * (1 / threshed_counts.sum())  # Use the reciprocal to avoid nan and dense matrix
        joint_probabilities = coo_matrix(joint_probabilities)
        marginal_x = joint_probabilities.sum(axis=1).flatten()
        marginal_y = joint_probabilities.sum(axis=0).flatten()

        return joint_probabilities, marginal_x, marginal_y

    def fit(self, sentences: Iterable[TokenList], **kwargs):
        for sentence in sentences:
            self._ingest_sentence(sentence)

    def predict(self, sentences: Iterable[TokenList], **kwargs):
        raise NotImplementedError

    def fit_predict(self, sentences: Iterable[TokenList], **kwargs):
        raise NotImplementedError

    def model_score(self):
        if self.normalized:
            return self.norm_mi_score()
        else:
            return self.mi_score()

    def mi_score(self):
        joint_probabilities, marginal_x, marginal_y = self._get_probabilities()
        mi = 0.0
        for i, xy_p in enumerate(joint_probabilities.data):
            x_ind = joint_probabilities.row[i]
            y_ind = joint_probabilities.col[i]
            x_p = marginal_x[0, x_ind]
            y_p = marginal_y[0, y_ind]
            pmi = np.log2(xy_p / (x_p * y_p))
            mi += pmi
        return mi

    def norm_mi_score(self):
        joint_probabilities, marginal_x, marginal_y = self._get_probabilities()

        dividend = 0.0
        divisor = 0.0

        for i, xy_p in enumerate(joint_probabilities.data):
            x_ind = joint_probabilities.row[i]
            y_ind = joint_probabilities.col[i]
            x_p = marginal_x[0, x_ind]
            y_p = marginal_y[0, y_ind]

            pmi = np.log2(xy_p / (x_p * y_p))
            dividend += xy_p * pmi
            divisor += -(xy_p * -np.log2(xy_p))

        return dividend / divisor

    def flush(self):
        self.w2i = defaultdict(lambda: len(self.w2i), {"<SOS>": 0, "<EOS>": 1})
        self.i2w = defaultdict(lambda: len(self.i2w), {v: k for k,v in self.w2i.items()})
        self.counts = dok_matrix((sys.maxsize, sys.maxsize))  # Initialise as empty
