from typing import List, Union, Iterator

from conllu.models import Token, TokenList

from src.utils.abstractclasses import SentenceMainProcessor


class SentenceAnalyzer(SentenceMainProcessor):
    """Sentence-level dependency length checker"""

    def __init__(
        self,
        count_root: bool = False,
        count_direction: bool = False,
        tokenwise_scores: bool = False,
    ):
        self.count_root = count_root
        self.count_direction = count_direction
        self.tokenwise_scores = tokenwise_scores

    def process_sentence(self, sentence: TokenList, **kwargs):
        dependency_lengths = []
        intervener_complexities = []
        heads_list = self.get_sentence_heads(sentence)

        for token in self.iter_sentence_tokens(sentence):
            dependency_lengths.append(self.get_token_dependency_length(token))
            intervener_complexities.append(
                self.get_token_intervener_complexity(heads_list, token)
            )

        sentence_data: dict = {}

        # Sentence meta-information
        sentence_data.update(
            {
                "id": sentence.metadata["sent_id"],
                "sentlen": self.get_sentence_length(sentence),
            }
        )

        if self.count_direction:
            sentence_data.update({
                "leftarcs": sum(1 for deplen in dependency_lengths if deplen < 0),
                "rightarcs": sum(1 for deplen in dependency_lengths if deplen >= 0)
            })

        # Add sum dependency length and ICM information
        if self.count_direction:
            sentence_data.update(
                {
                    "leftdeplen": sum(
                        abs(deplen) for deplen in dependency_lengths if deplen < 0
                    ),
                    "rightdeplen": sum(
                        abs(deplen) for deplen in dependency_lengths if deplen >= 0
                    ),
                }
            )
            sentence_data.update(
                {
                    "leftICM": sum(abs(ic) for ic in intervener_complexities if ic < 0),
                    "rightICM": sum(abs(ic) for ic in intervener_complexities if ic >= 0),
                }
            )

        sentence_data.update(
            {"sumdeplen": sum(abs(deplen) for deplen in dependency_lengths)}
        )
        sentence_data.update({"sumICM": sum(abs(ic) for ic in intervener_complexities)})

        # Keep tokenwise scores if enabled
        if self.tokenwise_scores and self.count_direction:
            sentence_data.update(
                {"deplens": dependency_lengths, "interveners": intervener_complexities}
            )

        elif self.tokenwise_scores:
            sentence_data.update(
                {
                    "deplens": list(abs(deplen) for deplen in dependency_lengths),
                    "interveners": list(abs(ic) for ic in intervener_complexities),
                }
            )

        return sentence_data

    def iter_sentence_tokens(self, tokenlist: TokenList):
        for token in tokenlist:
            if not self.count_root and token["head"] == 0:
                continue
            elif not isinstance(token["id"], int):
                continue
            elif not isinstance(token["head"], int):
                continue
            else:
                yield token

    def get_token_dependency_length(self, token: Token) -> int:
        token_position = token["id"]
        head_position = token["head"]
        dependency_length = token_position - head_position

        return dependency_length

    def get_token_intervener_complexity(self, heads_list: List, token: Token):
        dep_id = token["id"]
        head_id = token["head"]

        if dep_id < head_id:
            lower = dep_id + 1  # Must not include self
            upper = head_id + 1
            sign = -1  # Left branching
        else:
            lower = head_id
            upper = dep_id
            sign = 1  # Right branching

        n_intervening_heads: int = sum(
            1 for head_id in heads_list if head_id in range(lower, upper)
        )

        return n_intervening_heads * sign

    def get_sentence_length(self, sentence: TokenList) -> int:
        n_tokens = 0
        for token in sentence:
            if not isinstance(token["id"], int):
                continue
            n_tokens += 1

        return n_tokens

    def get_sentence_heads(self, tokenlist: Union[TokenList, Iterator[Token]]):
        heads = set()
        for token in tokenlist:
            heads.add(token["head"])
        return sorted(heads)
