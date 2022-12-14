from conllu.models import TokenList


class SentenceCleaner:
    def __init__(self, remove_fields: dict):
        self.remove_fields = remove_fields if isinstance(remove_fields, dict) else {}

    def __call__(self, tokenlist: TokenList):
        tokenlist = self.remove_nonstandard_tokens(tokenlist)
        tokenlist = self.remove_tokens(tokenlist)
        tokenlist = fix_tree_indices(tokenlist)
        return tokenlist

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


def make_index_mapping(tokenlist: TokenList) -> dict:
    index_mapping = {0: 0, None: None}
    i = 1
    for token in tokenlist:
        if isinstance(token["id"], int):
            index_mapping[token["id"]] = i
            i += 1
    return index_mapping


def fix_tree_indices(tokenlist: TokenList):
    new_tokenlist = tokenlist.copy()

    index_mapping = make_index_mapping(new_tokenlist)
    for i, token in enumerate(new_tokenlist):
        token_id = token["id"]
        token_head = token["head"]
        new_tokenlist[i]["id"] = index_mapping[token_id]
        new_tokenlist[i]["head"] = index_mapping[token_head]

    return new_tokenlist


def filter_preserve_metadata(tokenlist: TokenList, **kwargs):
    sentence = tokenlist.filter(**kwargs)
    sentence.metadata = tokenlist.metadata
    return sentence
