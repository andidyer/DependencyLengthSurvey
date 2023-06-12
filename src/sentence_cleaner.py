from typing import List, Dict, Tuple, AnyStr

from conllu.models import TokenList

from src.utils.abstractclasses import SentencePreProcessor
from src.utils.treeutils import standardize_deprels

from src.utils.decorators import (
    deepcopy_tokenlist,
    preserve_metadata,
    fix_token_indices,
)


class SentenceCleaner(SentencePreProcessor):
    def __init__(
        self,
        remove_config: List[Dict] = None,
        fields_to_empty: List[AnyStr] = None,
        mask_words: bool = False,
    ):
        self.remove_config = (
            [obj for obj in remove_config] if remove_config is not None else []
        )
        self.fields_to_empty = (
            fields_to_empty if isinstance(fields_to_empty, list) else []
        )
        self.mask_words = mask_words

    @deepcopy_tokenlist
    def process_sentence(self, sentence: TokenList, **kwargs):
        sentence = self.remove_nonstandard_tokens(sentence)
        if self.mask_words:
            sentence = self.mask_token_lexicon(sentence)
        sentence = self.remove_tokens(sentence)
        sentence = standardize_deprels(sentence)
        sentence = self.empty_fields(sentence)
        return sentence

    def remove_tokens(self, tokenlist: TokenList):

        remove_ids = set()

        # Find ids of tokens that would initially be removed by the filters
        for filter_item in self.remove_config:
            ids = (token["id"] for token in tokenlist.filter(**filter_item))
            remove_ids.update(ids)

        # While loop to find tokens in subtrees of removed items
        stack = list(remove_ids)  # Initialise as remove ids found at top level
        queue: List[Tuple] = [(token["id"], token["head"]) for token in tokenlist]

        while stack:

            new_ids = []
            for i, (dep_id, head_id) in enumerate(queue):
                if head_id in stack:
                    new_ids.append(dep_id)
            remove_ids.update(stack)
            stack = new_ids  # Terminate when there are no new ids
            queue = list((di, hi) for (di, hi) in queue if di not in stack)

        remove_ids = list(remove_ids)

        tokenlist = self.remove_tokens_by_id(tokenlist, remove_ids)

        return tokenlist

    def empty_fields(self, tokenlist: TokenList):
        new_tokenlist = tokenlist.copy()
        for i, token in enumerate(new_tokenlist):
            for field in self.fields_to_empty:
                if field not in new_tokenlist[i]:
                    raise KeyError(
                        f"Token does not contain field {field}; this cannot be emptied"
                    )
                else:
                    new_tokenlist[i][field] = "_"
        return new_tokenlist

    @staticmethod
    def remove_nonstandard_tokens(tokenlist: TokenList):
        """
        Removes any non-standard tokens, such as multi-word tokens or
        enhanced dependencies, from the sentence.
        """
        return tokenlist.filter(id=lambda x: isinstance(x, int))

    @staticmethod
    def remove_tokens_by_id(tokenlist: TokenList, ids: List[int]):
        return tokenlist.filter(id=lambda x: x not in ids)

    def mask_token_lexicon(self, tokenlist: TokenList):
        """Removes lexical information of the sentence. Replaces metadata text with ***MASKED***, and
        token form and lemma with [sent_id]+[token_id], e.g. 4-1, 4-2, ..., 4-n"""
        sent_id = tokenlist.metadata["sent_id"]
        tokenlist.metadata["text"] = "*MASKED*"
        for i, token in enumerate(tokenlist):
            token_id = token["id"]
            replace_value = f"f{sent_id}-{token_id}"
            tokenlist[i]["form"] = replace_value
            tokenlist[i]["lemma"] = replace_value

        return tokenlist
