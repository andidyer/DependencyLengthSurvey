import pytest
from conllu.models import Token, TokenList

from src.sentence_permuter import reorder_tokens


@pytest.fixture
def mapping():
    return {
        0: 0,
        1: 3,
        2: 2,
        3: 1,
        4: 7,
        5: 8,
        6: 4,
        7: 6,
        8: 5,
    }


@pytest.fixture
def sentence():
    return TokenList(
        [
            Token(id=1, form="Me", head=4),
            Token(id=2, form="and", head=3),
            Token(id=3, form="John", head=1),
            Token(id=4, form="go", head=0),
            Token(id=5, form="quickly", head=4),
            Token(id=6, form="to", head=8),
            Token(id=7, form="the", head=8),
            Token(id=8, form="library", head=4),
        ]
    )


@pytest.fixture
def expected_output():
    return TokenList(
        [
            Token(id=1, form="John", head=3),
            Token(id=2, form="and", head=1),
            Token(id=3, form="Me", head=7),
            Token(id=4, form="to", head=5),
            Token(id=5, form="library", head=7),
            Token(id=6, form="the", head=5),
            Token(id=7, form="go", head=0),
            Token(id=8, form="quickly", head=7),
        ]
    )


def test_reorder_tokens(mapping, sentence, expected_output):
    resVal = reorder_tokens(mapping, sentence)
    assert resVal == expected_output
