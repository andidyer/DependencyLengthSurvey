from typing import List, Dict, SupportsInt, AnyStr, Union

from conllu import Token, TokenList

from src.utils.decorators import (
    fix_token_indices,
    preserve_metadata,
    deepcopy_tokenlist,
)
from src.utils.abstractclasses import SentencePreProcessor
from src.utils.sentence_selector_functions import sentence_recursive_match
from src.utils.recursive_query import Query, make_query_from_dict

TOKEN_FIELDS = ("form", "lemma", "upos", "xpos", "deprel")


class SentenceSelector(SentencePreProcessor):
    def __init__(self, query: dict = None):
        if query is None:
            query = {}
        self.query = make_query_from_dict(query)

    @deepcopy_tokenlist
    def process_sentence(self, sentence: TokenList, **kwargs):
        matching_tokens = sentence_recursive_match(self.query, sentence)

        # TODO: The recursive match is currently broken and does not return dependant matches below the first level,
        #       though the absence of them still cause the query to fail as expected
        #       Until it is fixed, return the full sentence

        if len(matching_tokens) > 0:
            return sentence
        else:
            return TokenList()
