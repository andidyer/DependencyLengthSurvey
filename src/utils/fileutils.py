import json
from pathlib import Path
from typing import Union, Dict
import numpy as np

from conllu import TokenList
import logging


def load_ndjson(ndjson_file: Path):
    with open(ndjson_file, encoding="utf-8") as fin:
        for line in fin:
            yield json.loads(line.strip())


def serialize_data_item(data_item: Union[TokenList, Dict]):
    if isinstance(data_item, TokenList):
        return data_item.serialize()
    elif isinstance(data_item, dict):
        return json.dumps(data_item)


def load_word2vec(w2vfile: Path):

    w2v = {}

    with open(w2vfile, encoding="utf-8") as fin:
        header = fin.readline().strip()
        lines, dims = map(int, header.split())

        for i in range(lines):
            line = fin.readline()
            word, vec = line.strip().split(" ", maxsplit=1)
            vec = vec.split(" ")

            if len(vec) != dims:
                # This indicates a badly formatted line
                logging.warning(
                    f"Bad line {i+1} in {w2vfile}. {len(vec)} != {dims} Skipping this line"
                )
                continue

            vec = list(map(float), vec)
            vec = np.array(vec)

            w2v[word] = vec

    return w2v
