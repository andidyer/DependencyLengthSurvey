import json
from pathlib import Path
from typing import Union, Dict

from conllu import TokenList


def load_ndjson(ndjson_file: Path):
    with open(ndjson_file, encoding="utf-8") as fin:
        for line in fin:
            yield json.loads(line.strip())


def serialize_data_item(data_item: Union[TokenList, Dict]):
    if isinstance(data_item, TokenList):
        return data_item.serialize()
    elif isinstance(data_item, dict):
        return json.dumps(data_item)
