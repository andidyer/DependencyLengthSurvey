

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
usage: get_dependency_lengths.py [-h] [--treebank TREEBANK | --directory DIRECTORY] [--datafile DATAFILE] [--count_root] [--min_len MIN_LEN] [--max_len MAX_LEN]

optional arguments:
  -h, --help            show this help message and exit
  --count_root          Whether to count the dependency length of the root node
  --min_len MIN_LEN     Exclude sentences with less than a given minimum number of tokens
  --max_len MAX_LEN     Exclude sentences with more than a given maximum number of tokens

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
--datafile my_output_file.ndjson
```

Alternatively, if your data is in several treebanks, you can 
read from all files in a specified directory.

```shell
user$ python get_dependency_lengths.py \
--directory my_treebank_directory/ \
--datafile my_output_file.ndjson
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
--datafile my_output_file.ndjson \
--count_root
```

You can use the options `--min_len` and `--max_len` to exclude sentences that have 
fewer than a minimum or more than a maximum number of tokens. For example:

```shell
user$ python get_dependency_lengths.py \
--treebank my_treebank.conllu \
--datafile my_output_file.ndjson \
--min_len 5 \
--max_len 25
```

### The output

The output of the script is a file in ndjson format containing a 
list of json objects, where each object contains 
- `sentence_id`: The sentence ID (taken from the sentence metadata in the conllu file)
- `sentence_length`: The sentence length (in tokens, excluding those that have been excluded from the analysis)
- `sentence_sum_dependency_length`: The sum dependency length of the sentence (the sum of dependency lengths between all relation pairs in the sentence)
- `sentence_dependency_lengths`: A list of individual dependency lengths in the sentence (excluding those tokens excluded from the analysis)

This file should be named with the file extension `.ndjson`.

### Considerations

- The script is currently set up to ignore punctuation tokens,
as well as the root of the sentence. That is to say that the 
lengths of dependency from a punctuation token to its head and
from the token with head 0 are ignored.
- Any sentences with disjoint trees are currently ignored.
- Non-standard tokens such as enhanced dependencies and 
multiword tokens are ignored.

## Analysing dependency length data

Not yet implemented.

This script will analyse the sentence data generated by `get_dependency_lengths.py`.

## Permuting treebanks

Not yet implemented.

This script will produce permutations of treebanks according to set permutation functions.
These permutations will serve as baselines and counterfactuals, which we will use to 
determine whether observed dependency lengths in real corpora are systematically lower
than what might be expected by chance.

## TODO:
- [x] Option to toggle ignore root.
- [x] Option to filter sentences by max and min number of tokens
- [ ] Argument for filtering tokens based on fields
and features.
- [ ] Ability to filter multiple values within a field
- [ ] Option to toggle allow disjoint trees
- [ ] Start working on treebank permutation models
