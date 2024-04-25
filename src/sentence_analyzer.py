from typing import List, Union, Iterator
from dataclasses import dataclass, field

from conllu import Token, TokenList, TokenTree
import numpy as np

class SentenceAnalyzer:
    def __init__(
        self,
        token_analyzers: List[str] = None,
        w2v: dict = None,
        language: str = None,
        count_root: bool = False,
        aggregate=False,
    ):
        self._init_token_analyzers(
            token_analyzers, w2v=w2v, language=language, count_root=count_root
        )
        self.aggregate = aggregate

    def _init_token_analyzers(
        self,
        analyzers: List[str],
        w2v: dict = None,
        language: str = None,
        count_root: bool = False,
    ):
        self.token_analyzers = []
        if "DependencyLength" in analyzers:
            self.token_analyzers.append(DependencyLengthAnalyzer(count_root=count_root))
        if "IntervenerComplexity" in analyzers:
            self.token_analyzers.append(
                IntervenerComplexityAnalyzer(count_root=count_root)
            )
        if "HeadDependentDepth" in analyzers:
            self.token_analyzers.append(
                HeadDependentDepthAnalyzer(count_root=count_root)
            )
        if "SemanticSimilarity" in analyzers:
            self.token_analyzers.append(SemanticSimilarityAnalyzer(w2v))
        if "WordFrequency" in analyzers:
            self.token_analyzers.append(WordFrequencyAnalyzer(language))
        if "WordZipfFrequency" in analyzers:
            self.token_analyzers.append(WordZipfFrequencyAnalyzer(language))

    def process_sentence(self, sentence: TokenList):
        analyses = {analyzer.name: analyzer._process_tokens(sentence) for analyzer in self.token_analyzers}
        if self.aggregate:
            output_json = {
                "ID": sentence.metadata["sent_id"],
                "Length": len(sentence),
            }
            for analyzer_name, analysis_values in analyses.items():
                analysis_values = list(analysis_values)
                output_json.update({analyzer_name: np.abs(np.nansum(analysis_values)).item()})
            return output_json
        else:
            for analyzer_name, analysis_values in analyses.items():
                for i, (token, value) in enumerate(zip(sentence, analysis_values)):
                    if sentence[i]["misc"] is None:
                        sentence[i]["misc"] = {}
                    sentence[i]["misc"][analyzer_name] = value

            return sentence


class SentenceTokensAnalyzer:
    def _process_token(self, mapping: dict, token: Token):
        pass

    def process_sentence(self, sentence: TokenList, **kwargs):
        pass


class DependencyLengthAnalyzer(SentenceTokensAnalyzer):

    name = "DL"

    def __init__(self, count_root=False):
        self.count_root = count_root

    def _process_token(self, token: Token):
        if not self.count_root and token["head"] == 0:
            return 0
        else:
            distance = token["id"] - token["head"]
            return distance

    def _process_tokens(self, tokenlist: Union[TokenList, Iterator[Token]]):
        scores = []
        for token in tokenlist:
            if not isinstance(token["id"], int):
                continue
            scores.append(self._process_token(token))
        return scores

    def process_sentence(
        self, tokenlist: Union[TokenList, Iterator[Token]], aggregate=False
    ):
        scores = list(self._process_tokens(tokenlist))
        return scores


class IntervenerComplexityAnalyzer(SentenceTokensAnalyzer):

    name = "ICM"

    def __init__(self, count_root=False):
        self.count_root = count_root

    def _process_token(self, mapping: dict, token: Token):
        if not self.count_root and token["head"] == 0:
            return 0

        id, head = token["id"], token["head"]
        if id < head:
            lo, hi = id + 1, head
        else:
            lo, hi = head, id - 1

        sentence_heads = (
            item.token["id"] for key, item in mapping.items() if item.has_children()
        )
        n_intervening_heads = sum(1 for head in sentence_heads if lo <= head <= hi)
        return n_intervening_heads

    def _process_tokens(self, tokenlist: Union[TokenList, Iterator[Token]]):

        mapping = make_tokens_mapping(tokenlist)

        for token in tokenlist:
            if not isinstance(token["id"], int):
                continue

            yield self._process_token(mapping, token)

    def process_sentence(
        self, tokenlist: Union[TokenList, Iterator[Token]], aggregate=False
    ):
        scores = list(self._process_tokens(tokenlist))
        if not aggregate:
            return scores
        else:
            return np.sum(np.abs(scores)).item()


class HeadDependentDepthAnalyzer(SentenceTokensAnalyzer):

    name = "HDD"

    def __init__(self, count_root=False):
        self.count_root = count_root

    def _process_token(self, mapping: dict, token: Token):
        if token["deprel"] == "root":
            return 1 if self.count_root else 0
        else:
            head_token = mapping[token["head"]].token
            return self._process_token(mapping, head_token) + 1

    def _process_tokens(self, tokenlist: TokenList):
        mapping = make_tokens_mapping(tokenlist)
        result = list(self._process_token(mapping, token) for token in tokenlist)
        return result


    def process_sentence(self, sentence: TokenList, aggregate=False, **kwargs):
        pass

class SemanticSimilarityAnalyzer(SentenceTokensAnalyzer):

    name = "SemSim"

    def __init__(self, w2v: dict):
        self.w2v = w2v

    def word_cosine(self, word1, word2):
        word1vec = self.w2v[word1]
        word2vec = self.w2v[word2]
        return word1vec @ word2vec

    def _process_token(self, mapping: dict, token: Token):
        if token["head"] == 0:
            return np.nan

        token_form = token["form"]
        head_form = mapping[token["head"]].token["form"]

        return self.word_cosine(token_form, head_form)

    def _process_tokens(self, tokenlist: Union[TokenList, Iterator[Token]]):
        mapping = make_tokens_mapping(tokenlist)
        for token in tokenlist:
            if not isinstance(token["id"], int):
                continue
            elif token["head"] == 0:
                yield np.NaN
                continue
            else:
                token_form = token["form"]
                head_form = mapping[token["head"]].token["form"]

                yield self.word_cosine(token_form, head_form)

    def process_sentence(
        self, tokenlist: Union[TokenList, Iterator[Token]], aggregate=False
    ):
        scores = list(self._process_tokens(tokenlist))
        if aggregate:
            return np.nanmean(scores).item()
        else:
            return scores


class WordFrequencyAnalyzer(SentenceTokensAnalyzer):

    name = "Freq"

    def __init__(self, lang: str):
        raise NotImplementedError


class WordZipfFrequencyAnalyzer:

    name = "ZipfFreq"

    def __init__(self, lang: str):
        raise NotImplementedError

    def word_frequency(self, word: str):
        pass


@dataclass
class TokenExtra:
    token: Token
    head: int
    children: List[int] = field(default_factory=list)

    def has_children(self):
        return len(self.children) > 0


def make_tokens_mapping(sentence: TokenList) -> dict:
    root_token = Token(id=0, form="___root___")
    mapping = {0: TokenExtra(root_token, None)}
    for token in sentence:
        if not isinstance(token["id"], int):
            continue
        mapping[token["id"]] = TokenExtra(token, token["head"])

    for token in sentence:
        mapping[token["head"]].children.append(token["id"])

    return mapping
