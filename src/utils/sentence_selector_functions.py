from src.utils.recursive_query import Query
from conllu import Token, TokenList
from typing import List, Generator, Union
from dataclasses import dataclass, field


@dataclass
class TokenEntry:
    token: Token
    head: Union[int, None]
    children: List[int] = field(default_factory=list)

    def has_children(self):
        return len(self.children) > 0

    def is_root(self):
        return not isinstance(self.head, int)


class TokenMapping(dict):
    def get_token(self, id_):
        return self[id_].token

    def get_head_token(self, id_):
        head_id = self[id_].head
        return self.get_token(head_id)

    def get_child_tokens(self, id_):
        return [self.get_token(dep_id) for dep_id in self[id_].children]


def make_tokens_mapping(sentence: TokenList) -> TokenMapping:

    token_mapping = TokenMapping()
    token_mapping.update({0: TokenEntry(Token(id=0, form="___root___"), None)})

    for token in sentence:
        if not isinstance(token["id"], int):
            continue
        token_mapping[token["id"]] = TokenEntry(token, token["head"])

    for token in sentence:
        token_mapping[token["head"]].children.append(token["id"])

    return token_mapping


def _token_fields_match(query: Query, token: Token):

    query_token_fields = query.token_fields_dict()

    for field, value in query_token_fields.items():
        if value is None:
            pass
        elif field not in token:
            pass
        elif value != token[field]:
            return False

    return True


def _token_direction_match(query: Query, token: Token):
    if query.direction == 0:
        return True

    token_direction = 1 if token["id"] - token["head"] > 0 else -1
    if token_direction == query.direction:
        return True
    else:
        return False


def _query_match_dependants(
    mapping: TokenMapping, query: Query, dependants: List[Token]
):
    lo, hi = query.n_required
    matched_tokens = []

    for d_tok in dependants:
        if _token_recursive_match(mapping, query, d_tok):
            matched_tokens.append(d_tok)

            # If the query matches more than the maximum allowed number of tokens, break and return False
            if len(matched_tokens) > hi:
                return False

    # Return false if there are not enough matched tokens
    if len(matched_tokens) < lo:
        return False

    return matched_tokens


def _token_dependants_match(mapping: TokenMapping, query: Query, token: Token):
    dependant_tokens: List[Token] = mapping.get_child_tokens(token["id"])
    dependant_queries: List[Query] = query.dependants

    if dependant_queries is None:
        return None

    dependant_matches = []

    for dq in dependant_queries:
        query_match = _query_match_dependants(mapping, dq, dependant_tokens)
        if query_match is False:
            return False
        else:
            dependant_matches.append(query_match)

    return dependant_matches


def _token_head_match(mapping: TokenMapping, query: Query, token: Token):

    head_token = mapping[token["id"]].head
    head_query = query.head

    if head_query is None:
        return None

    head_match = _token_recursive_match(mapping, head_query, head_token)

    lo, hi = head_query.n_required

    if lo <= len(head_match) <= hi:
        return head_match
    else:
        return False


def _token_recursive_match(mapping: TokenMapping, query: Query, token: Token):

    matching_tokens = []

    if query is None:
        return matching_tokens

    # 1. Check fields match
    if not _token_fields_match(query, token):
        return matching_tokens

    # 2. Check direction match
    if not _token_direction_match(query, token):
        return matching_tokens

    # 3. Check head match
    head_match = _token_head_match(mapping, query, token)
    if head_match is False:
        return matching_tokens

    # 4. Check dependants match
    dependants_match = _token_dependants_match(mapping, query, token)
    if dependants_match is False:
        return matching_tokens

    matching_tokens.append(token)

    if head_match is not None:
        matching_tokens.append(head_match)

    if dependants_match is not None:
        matching_tokens.append(dependants_match)

    return matching_tokens


def flatten_nested_list(nested_list) -> Generator:
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_nested_list(item)
        else:
            yield item


def token_recursive_match(mapping: TokenMapping, query: Query, token: Token):

    """
    :param mapping: A mapping of the relations between tokens
    :param query: A query object
    :param token: A conllu token object
    :return: A list of unique matched tokens
    """

    output_tokens = []

    matched_tokens_nested = _token_recursive_match(mapping, query, token)
    matched_tokens_generator = flatten_nested_list(matched_tokens_nested)

    for tok in matched_tokens_generator:
        if tok not in output_tokens:
            output_tokens.append(tok)

    return output_tokens


def sentence_recursive_match(query: Query, tokenlist: TokenList):

    mapping = make_tokens_mapping(tokenlist)

    output_tokens = []

    for token in tokenlist:
        result = token_recursive_match(mapping, query, token)
        if result:
            for token in result:
                if token not in output_tokens:
                    output_tokens.append(token)

    return output_tokens
