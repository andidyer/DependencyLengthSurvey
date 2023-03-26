from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Union


@dataclass(frozen=True)
class Query:
    form: str = None
    lemma: str = None
    upos: str = None
    xpos: str = None
    deprel: str = None

    n_required: tuple = (1,1)
    direction: int = 0

    head: Query = None
    dependants: List[Query] = None

    _isQueryRoot: bool = field(default=True, repr=False)
    _isHeadQuery: bool = field(default=False, repr=False)
    _isDependantQuery: bool = field(default=False, repr=False)

    def token_fields_dict(self):
        return {field: value for (field, value) in asdict(self).items() if field in ("form", "lemma", "upos", "xpos", "deprel")}

    def isQueryRoot(self):
        return self._isQueryRoot

    def isHeadQuery(self):
        return self._isHeadQuery

    def isDependantQuery(self):
        return self._isDependantQuery


def make_query_from_dict(query_json: dict):
    query_dict = {}
    for field, value in query_json.items():
        if field == "n_required":
            query_dict[field] = parse_n_required(value)
        elif field == "head":
            query_dict[field] = _parse_head_query(value)
        elif field == "dependants":
            query_dict[field] = _parse_dependant_queries(value)
        else:
            query_dict[field] = value

    return Query(**query_dict)


def parse_n_required(n_required: Union[str, int, List[int]]):
    if n_required == "?":
        return (0, 1)
    elif n_required == "+":
        return (1, 255)
    elif n_required == "*":
        return (0, 255)
    elif isinstance(n_required, int):
        return (n_required,) * 2
    elif isinstance(n_required, list) and len(n_required) == 2 and all(isinstance(element, int) for element in n_required):
        return tuple(n_required)
    else:
        raise ValueError("""
            n_required parameter must follow one of the following patterns:
            int, [int, int], '*', '+', '?'
        """)


def _parse_head_query(head_query: dict):
    head_query["_isQueryRoot"] = False
    head_query["_isHeadQuery"] = True
    return make_query_from_dict(head_query)


def _parse_dependant_queries(dependant_queries: List[dict]):
    return list(_parse_dependant_query(query) for query in dependant_queries)


def _parse_dependant_query(dependant_query: dict):
    dependant_query["_isQueryRoot"] = False
    dependant_query["_isDependantQuery"] = True
    return make_query_from_dict(dependant_query)
