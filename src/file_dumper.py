import logging
from pathlib import Path
from typing import Iterable

from src.utils.fileutils import serialize_data_item


class FileDumper:
    def __init__(self, extension: str):
        self.extension = extension

    def write_to_file(self, data_stream: Iterable, outfile: Path):
        with open(outfile, "w", encoding="utf-8") as fout:
            for data_item in data_stream:
                serialized_item = serialize_data_item(data_item)
                print(serialized_item, file=fout)

    def make_equivalent_paths(self, source_dir: Path, infile: Path, outdir: Path):
        """Creates the fullpath to the file and outputs"""
        # Get path of infile relative to parent path
        infile_relpath = Path(infile).relative_to(source_dir)

        # Get the parent directory of the file
        parent = infile_relpath.parent

        stem = infile_relpath.stem

        # Put together the output directory and the new file path
        outfile_parent = Path(outdir, parent)

        outfile_name = Path(f"{stem}{self.extension}")

        if not outfile_parent.exists():
            logging.info(f"Making parent path: {outfile_parent}")
            outfile_parent.mkdir(parents=True)

        return Path(outfile_parent, outfile_name)
