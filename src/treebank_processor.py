from typing import List, Dict, Union

from conllu import TokenList, SentenceList

from src.utils.abstractclasses import SentenceMainProcessor


class TreebankProcessor:
    """ "
    Class for performing an operation on a treebank.

    Uses an inner function, whici
    """

    def __init__(self, sentence_processor: SentenceMainProcessor):
        self.sentence_processor = sentence_processor
        self.fileext = self.sentence_processor.fileext

    def process_treebank(self, treebank: SentenceList, **kwargs):
        """ "Reads a set of conllu files and executes the function on them."""
        processed_sentences: List[Union[TokenList, Dict]] = []
        for sentence in treebank:
            processed_sentences.append(
                self.sentence_processor.process_sentence(sentence, **kwargs)
            )
        return processed_sentences
