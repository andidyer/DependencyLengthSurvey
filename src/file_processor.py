import json

from typing import Union, Dict, Iterable
from pathlib import Path
from conllu import TokenList, SentenceList

from src.treebank_processor import TreebankProcessor
from src.load_treebank import TreebankLoader


class FileProcessor:
    """ "
    Takes in a file or set of files and processes them with the following steps:
    - loading and cleaning
    - analysis OR permutation
    - dumping to file(s)
    """

    def __init__(self, loader: TreebankLoader, treebank_processor: TreebankProcessor):
        self.loader = loader
        self.treebank_processor = treebank_processor

    def load_conllu_file(self, infile: Path):
        return self.loader.load_treebank(infile)

    def process_treebank(self, treebank: SentenceList):
        return self.treebank_processor.process_treebank(treebank)

    @staticmethod
    def serialize_data(data_item: Union[TokenList, Dict]):
        """ "Utility function for serializing the data to a printable form
        :param data_item: Either a TokenList sentence or a json-serializable dictionary
        :return: Either a serialized conllu format sentence or a json string format dictionary
        """
        if isinstance(data_item, dict):
            return json.dumps(data_item)
        elif isinstance(data_item, TokenList):
            return data_item.serialize()
        else:
            raise TypeError(
                f"""
            Unrecognized data output type. Object type must be:
                - dict
                - conllu.TokenList
            Received type {type(data_item)}
                """
            )

    def dump_to_file(self, data: Union[SentenceList, Iterable[Dict]], outfile: Path):
        with open(outfile, "w", encoding="utf-8") as fout:
            # Needs to do follow the logic of either analyzer csv dumping or treebank dumping
            for data_item in data:
                serialized = self.serialize_data(data_item)
                print(serialized, file=fout)

    def process_file(self, infile: Path, outfile: Path):
        """Main function for processing a single file"""
        input_treebank = self.load_conllu_file(infile)
        processed_treebank = self.process_treebank(input_treebank)
        self.dump_to_file(processed_treebank, outfile)
