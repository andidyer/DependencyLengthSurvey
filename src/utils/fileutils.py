import json
import csv
from pathlib import Path
from typing import Iterable, Union
from conllu import Token, TokenList, TokenTree, SentenceList


class FileDumper:
    # DEPRECATED
    @staticmethod
    def dump_treebank_data_as_json(treebank_data: Iterable[dict], outfile: Path):
        with open(outfile, "w", encoding="utf-8") as fout:
            json.dump(list(treebank_data), fout, ensure_ascii=False, indent=1)

    @staticmethod
    def dump_treebank_data_as_ndjson(treebank_data: Iterable[dict], outfile: Path):
        with open(outfile, "w", encoding="utf-8") as fout:
            for obj in treebank_data:
                print(json.dumps(obj, ensure_ascii=False), file=fout)

    @staticmethod
    def dump_treebank_data_as_csv(treebank_data: Iterable[dict], outfile: Path):
        with open(outfile, "w", encoding="utf-8", newline="") as fout:
            fieldnames = treebank_data[0].keys()
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writerows(treebank_data)

    @staticmethod
    def dump_treebank_as_conllu(
        outfile: Path, treebank: Union[Iterable[TokenList], SentenceList]
    ):
        with open(outfile, "w", encoding="utf-8") as fout:
            for sentence in treebank:
                print(sentence.serialize(), file=fout)


def load_ndjson(ndjson_file: Path):
    with open(ndjson_file, encoding="utf-8") as fin:
        for line in fin:
            yield json.loads(line.strip())
