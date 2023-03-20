from conllu import TokenList
from functools import wraps, singledispatch
import copy
from typing import Callable

from src.utils.treeutils import make_index_mapping
from src.utils.abstractclasses import SentenceProcessor


@singledispatch
def deepcopy_tokenlist(function: Callable):
    @wraps(function)
    def inner(self: SentenceProcessor, tokenlist: TokenList, **kwargs):
        new_tokenlist = copy.deepcopy(tokenlist)
        new_tokenlist = function(self, new_tokenlist, **kwargs)

        return new_tokenlist

    return inner


def preserve_metadata(function: Callable):
    """
    Decorator for any functions that modify TokenList objects, so that they can retain metadata

    :param function: A function that modifies a TokenList
    :return: TokenList object with metadata
    """

    @wraps(function)
    def inner(self: SentenceProcessor, tokenlist: TokenList, **kwargs):
        new_tokenlist: TokenList = function(self, tokenlist, **kwargs)
        new_tokenlist.metadata = tokenlist.metadata

        return new_tokenlist

    return inner


def fix_token_indices(function: Callable):
    @wraps(function)
    def inner(self: SentenceProcessor, tokenlist: TokenList, **kwargs):

        tokenlist = function(self, tokenlist, **kwargs)

        # Fixes token indices after performing the function
        tokenlist = _fix_token_indices(tokenlist)

        return tokenlist

    return inner


def _fix_token_indices(tokenlist: TokenList):

    index_mapping = make_index_mapping(tokenlist)
    for i, token in enumerate(tokenlist):
        token_id = token["id"]
        token_head = token["head"]
        tokenlist[i]["id"] = index_mapping[token_id]
        tokenlist[i]["head"] = index_mapping[token_head]

    return tokenlist
