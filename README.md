# Dependency Length Survey

This code is used in our survey of dependency length in a parallel 
multilingual corpus. It is part of a replication study of 
[Futrell and Gibson (2015)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4547262/), 
but we expect the code to be extended for other, similar projects, and we will
add new scripts and modules to it periodically.

With this code you can:

- Permute input treebanks according to a permutation model,
  producing conllu files with the permuted output.
- Analyse sentences in treebanks for properties such as 
  dependency lenght, intervener complexity, and more.
- Perform both these steps in one go.


## Setting up the project.

You should first install the requirements necessary for these
scripts using the `pip` command and the provided
`requirements.txt`.

```commandline
user$ pip install -r requirements.txt
```

## Scripts

### analyze_treebanks.py

```
usage: analyze_treebanks.py [-h] [--treebank TREEBANK | --directory DIRECTORY]                                                                                     
                            [--analysis_modes {DependencyLength,IntervenerComplexity,SemanticSimilarity,WordFrequency,WordZipfFrequency} [{DependencyLength,IntervenerComplexity,SemanticSimilarity,WordFrequency,WordZipfFrequency} ...]]
                            [--glob_pattern GLOB_PATTERN] [--random_seed RANDOM_SEED] [--outfile OUTFILE | --outdir OUTDIR] [--remove_config REMOVE_CONFIG]
                            [--fields_to_remove [{form,lemma,upos,xpos,feats,deps,misc} [{form,lemma,upos,xpos,feats,deps,misc} ...]]] [--mask_words] [--min_len MIN_LEN]
                            [--max_len MAX_LEN] [--count_root] [--language LANGUAGE] [--aggregate] [--verbose]

optional arguments:
  -h, --help            show this help message and exit

required arguments:
  --treebank TREEBANK   Treebank to load and permute
  --directory DIRECTORY
                        Directory from which to find treebanks by globbing
  --analysis_modes {DependencyLength,IntervenerComplexity,SemanticSimilarity,WordFrequency,WordZipfFrequency} [{DependencyLength,IntervenerComplexity,SemanticSimilarity,WordFrequency,WordZipfFrequency} ...]
                        Metrics with which to analyse tokens/sentences
  --outfile OUTFILE     The file to output the permuted treebank(s) to
  --outdir OUTDIR       The directory to output the permuted treebank(s) to

optional arguments:
  --glob_pattern GLOB_PATTERN
                        glob pattern for recursively finding files that match the pattern
  --min_len MIN_LEN     Exclude sentences with less than a given minimum number of tokens
  --max_len MAX_LEN     Exclude sentences with more than a given maximum number of tokens
  --count_root          Include the root node in the sentence analysis
  --language LANGUAGE   Language (ISO639-1) for the WordFrequency analyzer
  --aggregate           If true, token scores will be aggregated and the results for each sentence will be output in an ndjson
  --verbose             Verbosity


```

### permute_treebanks.py

```
usage: permute_treebanks.py [-h] [--treebank TREEBANK | --directory DIRECTORY] [--glob_pattern GLOB_PATTERN] [--random_seed RANDOM_SEED]
                            [--permutation_mode {random_projective,random_same_valency,random_same_side,optimal_projective,original_order,fixed_order}]
                            [--outfile OUTFILE | --outdir OUTDIR] [--remove_config REMOVE_CONFIG]
                            [--fields_to_remove [{form,lemma,upos,xpos,feats,deps,misc} [{form,lemma,upos,xpos,feats,deps,misc} ...]]] [--n_times N_TIMES | --grammars GRAMMARS]        
                            [--min_len MIN_LEN] [--max_len MAX_LEN] [--mask_words] [--verbose]

optional arguments:
  -h, --help            show this help message and exit

required arguments:
  --treebank TREEBANK   Treebank to load and permute
  --directory DIRECTORY
                        Directory from which to find treebanks by globbing
  --permutation_mode {random_projective,random_same_valency,random_same_side,optimal_projective,original_order,fixed_order}
                        The type of permutation to perform
  --outfile OUTFILE     The file to output the permuted treebank(s) to
  --outdir OUTDIR       The directory to output the permuted treebank(s) to

optional arguments:
  --glob_pattern GLOB_PATTERN
                        glob pattern for recursively finding files that match the pattern
  --random_seed RANDOM_SEED
                        Random seed for permutation
  --remove_config REMOVE_CONFIG
                        ndjson format list of token properties to exclude
  --fields_to_remove [{form,lemma,upos,xpos,feats,deps,misc} [{form,lemma,upos,xpos,feats,deps,misc} ...]]
                        Masks any fields in a conllu that are not necessary; can save some space
  --n_times N_TIMES     Number of times to perform the permutation action on each treebank
  --grammars GRAMMARS   Number of times to perform the permutation action on each treebank
  --min_len MIN_LEN     Exclude sentences with less than a given minimum number of tokens
  --max_len MAX_LEN     Exclude sentences with more than a given maximum number of tokens
  --mask_words          Mask all words in the treebank. Token forms and lemma will be represented only by original token index.
  --verbose             Verbosity

```

### permute_and_analyze_treebanks.py

```
usage: permute_and_analyze_treebanks.py [-h] [--treebank TREEBANK | --directory DIRECTORY] [--glob_pattern GLOB_PATTERN] [--random_seed RANDOM_SEED]
                                        [--permutation_mode {random_projective,random_same_valency,random_same_side,optimal_projective,original_order,fixed_order}]
                                        [--outfile OUTFILE | --outdir OUTDIR] [--remove_config REMOVE_CONFIG]
                                        [--fields_to_remove [{form,lemma,upos,xpos,feats,deps,misc} [{form,lemma,upos,xpos,feats,deps,misc} ...]]]
                                        [--n_times N_TIMES | --grammars GRAMMARS] [--min_len MIN_LEN] [--max_len MAX_LEN] [--count_root] [--count_direction] [--tokenwise_scores]       
                                        [--verbose]

optional arguments:
  -h, --help            show this help message and exit

required arguments:
  --treebank TREEBANK   Treebank to load and permute
  --directory DIRECTORY
                        Directory from which to find treebanks by globbing
  --permutation_mode {random_projective,random_same_valency,random_same_side,optimal_projective,original_order,fixed_order}
                        The type of permutation to perform
  --outfile OUTFILE     The file to output the permuted treebank(s) to
  --outdir OUTDIR       The directory to output the permuted treebank(s) to

optional arguments:
  --glob_pattern GLOB_PATTERN
                        glob pattern for recursively finding files that match the pattern
  --random_seed RANDOM_SEED
                        Random seed for permutation
  --remove_config REMOVE_CONFIG
                        ndjson format list of token properties to exclude
  --fields_to_remove [{form,lemma,upos,xpos,feats,deps,misc} [{form,lemma,upos,xpos,feats,deps,misc} ...]]
                        Masks any fields in a conllu that are not necessary; can save some space
  --n_times N_TIMES     Number of times to perform the permutation action on each treebank
  --grammars GRAMMARS   Number of times to perform the permutation action on each treebank
  --min_len MIN_LEN     Exclude sentences with less than a given minimum number of tokens
  --max_len MAX_LEN     Exclude sentences with more than a given maximum number of tokens
  --count_root          Include the root node in the sentence analysis
  --count_direction     Count left and right branching dependencies separately
  --tokenwise_scores    Keep scores of all tokens in a separate field. This is a variable length list. If --count_direction is enabled, then left scores will have a negative sign      
  --verbose             Verbosity
```
