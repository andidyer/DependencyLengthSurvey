

# Dependency Length Survey

This is the working code for our project measuring sentence
dependency lengths across comparable corpora.

## Setting up the project.

You should first install the requirements necessary for these
scripts using the `pip` command and the provided 
`requirements.txt`.

```commandline
user$ pip install -r requirements.txt
```

## Getting dependency length data

### Running the script

The main script for getting dependency 
length data is `get_dependency_lengths.py`. Given a treebank
or directory of treebanks in `conllu` format, this script will 
read the treebank(s) and, for each sentence, produce a json 
object containing data about the sentence.

This script can be run as follows:

```text
usage: get_dependency_lengths.py [-h] [--treebank TREEBANK | --directory DIRECTORY] [--datafile DATAFILE] [--count_root]

optional arguments:
  -h, --help            show this help message and exit
  --count_root          Whether to count the dependency length of the root node

required arguments:
  --treebank TREEBANK   Treebank to load and analyse
  --directory DIRECTORY
                        Directory to load and analyse
  --datafile DATAFILE   File to output the per-sentence statistics json to

```
For example:

```shell
user$ python get_dependency_lengths.py \
--treebank my_treebank.conllu \
--datafile my_output_file.json
```

Alternatively, if your data is in several treebanks, you can 
read from all files in a specified directory.

```shell
user$ python get_dependency_lengths.py \
--directory my_treebank_directory/ \
--datafile my_output_file.json
```

Note that `--treebank` and `--directory` are mutually exclusive.
Whichever you use, the data will be dumped to a single file in
json format specified by `--datafile`.
For this reason, it is a good idea to ensure that in the case 
of multiple conllu files each individual sentence has a unique
identifier.

Adding the optional flag `--count_root` will ensure that 
dependency lengths from the root word to the root token will
also be counted.
This can be used as follows:

```shell
user$ python get_dependency_lengths.py \
--treebank my_treebank.conllu \
--datafile my_output_file.json \
--count_root
```

### The output

The output of the script is a file in json format containing a 
list of json objects, where each object contains 
- `sentence_id`: The sentence ID (taken from the sentence metadata in the conllu file)
- `sentence_length`: The sentence length (in tokens, excluding those that have been excluded from the analysis)
- `sentence_sum_dependency_length`: The sum dependency length of the sentence (the sum of dependency lengths between all relation pairs in the sentence)
- `sentence_dependency_lengths`: A list of individual dependency lengths in the sentence (excluding those tokens excluded from the analysis)

This file should be named with the file extension `.json`.

### Considerations

- The script is currently set up to ignore punctuation tokens,
as well as the root of the sentence. That is to say that the 
lengths of dependency from a punctuation token to its head and
from the token with head 0 are ignored.
- Any sentences with disjoint trees are currently ignored.
- Non-standard tokens such as enhanced dependencies and 
multiword tokens are ignored.

## TODO:
- [x] Option to toggle ignore root.
- [ ] Argument for filtering tokens based on fields
and features.
- [ ] Option to toggle allow disjoint trees
- [ ] Start working on treebank permutation models
