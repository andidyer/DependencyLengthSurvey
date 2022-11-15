import statistics as stats
import json
import csv
from typing import Iterable
from pathlib import Path


class TreebankStatistics:
    """Class for aggregating statistics"""

    @staticmethod
    def get_treebank_statistics(treebank_data: Iterable[dict]):
        return {
            "Average dependency length rate (d/l^2)": TreebankStatistics._get_avg_dependency_length_rate(
                treebank_data
            ),
        }

    @staticmethod
    def dump_aggregate_data_as_json(treebank_data: Iterable[dict], outfile: Path):
        aggregate_statistics = TreebankStatistics.get_treebank_statistics(treebank_data)
        with open(outfile, "w", encoding="utf-8") as fout:
            json.dump(aggregate_statistics, fout, ensure_ascii=False, indent=1)

    @staticmethod
    def _get_avg_dependency_length_rate(treebank_data: Iterable[dict]):
        dependency_length_rate: int = 0
        n = 0
        for i, data_obj in enumerate(treebank_data):
            dependency_length_rate += (
                data_obj["sentence_sum_dependency_length"]
                / data_obj["sentence_length"] ** 2
            )
            n += 1
        dependency_length_rate /= n

        return dependency_length_rate

    @staticmethod
    def _get_avg_dependency_length_deviation_rate(treebank_data: Iterable[dict]):
        dependency_length_deviation_rate = 0
        n = 0
        for i, data_obj in enumerate(treebank_data):
            dependency_length_deviation_rate += (
                stats.stdev(data_obj["sentence_dependency_lengths"])
                / data_obj["sentence_length"]
            )
            n += 1
        dependency_length_deviation_rate /= n

        return dependency_length_deviation_rate

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
