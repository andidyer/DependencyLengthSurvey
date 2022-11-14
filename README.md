

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

```commandline
user$ python get_dependency_lengths.py \
--treebank [TREEBANK: The path to the treebank to read] \
--datafile [DATAFILE: The path to the file where the statistics should be dumped]
```

Alternatively, if your data is in several treebanks, you can 
read from all files in a specified directory.

```commandline
user$ python get_dependency_lengths.py \
--directory [DIRECTORY: The path to the directory to read] \
--datafile [DATAFILE: The path to the file where the statistics should be dumped]
```

Note that `--treebank` and `--directory` are mutually exclusive.
Whichever you use, the data will be dumped to a single file in
json format specified by `--datafile`.
For this reason, it is a good idea to ensure that in the case 
of multiple conllu files each individual sentence has a unique
identifier.

### The output

The output of the script is a file in json format containing a 
list of json objects, where each object contains 
- `sentence_id`: The sentence ID (taken from the sentence metadata in the conllu file)
- `sentence_length`: The sentence length (in tokens, excluding those that have been excluded from the analysis)
- `sentence_sum_dependency_length`: The sum dependency length of the sentence (the sum of dependency lengths between all relation pairs in the sentence)
- `sentence_dependency_lengths`: A list of individual dependency lengths in the sentence (excluding those tokens excluded from the analysis)

This file should be named with the file extension `.json`.
