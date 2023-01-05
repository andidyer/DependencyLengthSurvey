from conllu.models import TokenList
from src.utils.abstractclasses import SentencePreProcessor
from src.utils.treeutils import fix_token_indices
from typing import List, Dict, Tuple
import logging


class SentenceCleaner(SentencePreProcessor):
    def __init__(self, remove_config: List[Dict]):
        self.remove_config = remove_config if isinstance(remove_config, list) else []

    def __call__(self, sentence: TokenList):
        return self.process_sentence(sentence)

    def process_sentence(self, sentence: TokenList, **kwargs):
        sentence = self.remove_nonstandard_tokens(sentence)
        sentence = self.remove_tokens(sentence)
        sentence = fix_token_indices(sentence)
        return sentence

    def remove_tokens(self, tokenlist: TokenList):

        remove_ids = set()

        # Find ids of tokens that would initially be removed by the filters
        for filter_item in self.remove_config:
            ids = (token["id"] for token in tokenlist.filter(**filter_item))
            remove_ids.update(ids)

        # While loop to find tokens in subtrees of removed items
        stack = list(remove_ids)    # Initialise as remove ids found at top level
        queue: List[Tuple] = [(token["id"], token["head"]) for token in tokenlist]

        while stack:

            new_ids = []
            for i, (dep_id, head_id) in enumerate(queue):
                if head_id in stack:
                    new_ids.append(dep_id)
            remove_ids.update(stack)
            stack = new_ids     # Terminate when there are no new ids
            queue = list((di, hi) for (di, hi) in queue if di not in stack)

        tokenlist = filter_preserve_metadata(
            tokenlist, id=lambda x: x not in remove_ids
        )

        return tokenlist

    @staticmethod
    def remove_nonstandard_tokens(tokenlist: TokenList):
        """
        Removes any non-standard tokens, such as multi-word tokens or
        enhanced dependencies, from the sentence.
        """
        return filter_preserve_metadata(tokenlist, id=lambda x: isinstance(x, int))


def filter_preserve_metadata(tokenlist: TokenList, **kwargs):
    sentence = tokenlist.filter(**kwargs)
    sentence.metadata = tokenlist.metadata
    return sentence
