from conllu.models import TokenList
from src.utils.abstractclasses import SentencePreProcessor
from src.utils.treeutils import fix_token_indices


class SentenceCleaner(SentencePreProcessor):
    def __init__(self, remove_fields: dict):
        self.remove_fields = remove_fields if isinstance(remove_fields, dict) else {}

    def __call__(self, sentence: TokenList):
        return self.process_sentence(sentence)

    def process_sentence(self, sentence: TokenList, **kwargs):
        sentence = self.remove_nonstandard_tokens(sentence)
        sentence = self.remove_tokens(sentence)
        sentence = fix_token_indices(sentence)
        return sentence

    def remove_tokens(self, tokenlist: TokenList):
        _remove_ids = tuple(
            token["id"] for token in tokenlist.filter(**self.remove_fields)
        )
        for idi in _remove_ids:
            _remove_ids += self._remove_tokens_helper(tokenlist, idi, _remove_ids)
        tokenlist = filter_preserve_metadata(
            tokenlist, id=lambda x: x not in _remove_ids
        )

        return tokenlist

    def _remove_tokens_helper(
        self, tokenlist: TokenList, parent_id: int, _remove_ids: tuple
    ):
        """Recursive function to remove all tokens where the id or parent is in a list of removable items"""
        child_ids = tuple(token["id"] for token in tokenlist.filter(head=parent_id))
        for idi in child_ids:
            _remove_ids += self._remove_tokens_helper(tokenlist, idi, _remove_ids)
        return child_ids

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
