import pytest
from conllu.models import Token, TokenList

from src.sentence_cleaner import SentenceCleaner


@pytest.fixture
def sentence():
    return TokenList(
        [
            Token(id=1, form="i", head=2, deprel="nsubj", upos="PRON"),
            Token(id=(2, "-", 3), form="wanna", head=None, deprel="root", pos="VERB"),
            Token(id=2, form="wan", head=0, deprel="root", upos="VERB"),
            Token(id=3, form="na", head=4, deprel="mark", upos="PART"),
            Token(id=4, form="meet", head=2, deprel="xcomp", upos="VERB"),
            Token(id=5, form="people", head=4, deprel="obj", upos="NOUN"),
            Token(id=6, form="from", head=7, deprel="case", upos="ADP"),
            Token(id=7, form="germany", head=5, deprel="nmod", upos="PROPN"),
            Token(id=8, form="i", head=11, deprel="nsubj", upos="PRON"),
            Token(id=9, form="am", head=11, deprel="cop", upos="AUX"),
            Token(id=10, form="from", head=11, deprel="case", upos="ADP"),
            Token(id=11, form="vietnam", head=2, deprel="parataxis", upos="PROPN"),
            Token(id=12, form=".", head=2, deprel="punct", upos="PUNCT"),
        ]
    )


@pytest.fixture
def expected_output1():
    return TokenList(
        [
            Token(id=1, form="i", head=2, deprel="nsubj", upos="PRON"),
            Token(id=2, form="wan", head=0, deprel="root", upos="VERB"),
            Token(id=3, form="na", head=4, deprel="mark", upos="PART"),
            Token(id=4, form="meet", head=2, deprel="xcomp", upos="VERB"),
            Token(id=5, form="people", head=4, deprel="obj", upos="NOUN"),
            Token(id=6, form="from", head=7, deprel="case", upos="ADP"),
            Token(id=7, form="germany", head=5, deprel="nmod", upos="PROPN"),
            Token(id=8, form="i", head=11, deprel="nsubj", upos="PRON"),
            Token(id=9, form="am", head=11, deprel="cop", upos="AUX"),
            Token(id=10, form="from", head=11, deprel="case", upos="ADP"),
            Token(id=11, form="vietnam", head=2, deprel="parataxis", upos="PROPN"),
        ]
    )


@pytest.fixture
def expected_output2():
    return TokenList(
        [
            Token(id=1, form="i", head=2, deprel="nsubj", upos="PRON"),
            Token(id=2, form="wan", head=0, deprel="root", upos="VERB"),
            Token(id=3, form="na", head=4, deprel="mark", upos="PART"),
            Token(id=4, form="meet", head=2, deprel="xcomp", upos="VERB"),
            Token(id=5, form="people", head=4, deprel="obj", upos="NOUN"),
            Token(id=6, form="from", head=7, deprel="case", upos="ADP"),
            Token(id=7, form="germany", head=5, deprel="nmod", upos="PROPN"),
            Token(id=8, form=".", head=2, deprel="punct", upos="PUNCT"),
        ]
    )


@pytest.fixture
def cleaner1():
    return SentenceCleaner([{"upos": "PUNCT"}])


@pytest.fixture
def cleaner2():
    return SentenceCleaner([{"deprel": "parataxis"}])


def test_clean_sentence1(cleaner1, sentence, expected_output1):
    resVal = cleaner1(sentence)
    assert resVal == expected_output1


def test_clean_sentence2(cleaner2, sentence, expected_output2):
    resVal = cleaner2(sentence)
    assert resVal == expected_output2
