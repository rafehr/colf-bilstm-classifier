# colf_bilstm_classifier

## BiLSTM-MLP classifier for the disambiguation of verbal idioms

The classifier as well as the [corpus of German VIDs](https://github.com/rafehr/COLF-VID) are described in more detail in the paper [Supervised Disambiguation of German Verbal Idioms with a BiLSTM Architecture](https://www.aclweb.org/anthology/2020.figlang-1.29.pdf).

### Requirements

### Training the classifier

There are two diffrent 'versions' of the classifier, since ELMo embeddings required some changes to the original architecture.

1. **Bilstm-mlp**

To train the classifier the COLF-VID_1.0 corpus is required (which is already included in `data/`).

The first step is to build the data set from the COLF-VID corpus:

```
python build_dataset.py
```

This performs a train/val/test split. For each split four different text files are created: `tokens.txt`, `lemmas.txt`, `pos.txt` and `labels.txt`. As the names of the files suggest, these contain the tokens, lemmas, POS tags and labels of each sentence in the corpus with one line per sentence. 

The next step is to remove the preposition labels from the `labels.txt` files:

```
python fetch_noun_verb_labels.py
```

This is done because not all VID types contain prepositions and the multilayer perceptron (MLP) - which is fed with the concatenated embeddings of the VID components - requires its input to be fixed length.

After that the vocabs (i.e. sets) of tokens, lemmas, POS tags and labels are build with:

```
python build_vocab.py
```

Furthermore dataset parameters are saved to a json file by this script.

After these steps the training is performed with:

```
python train.py
```

To train the classifier a text file of pretrained embeddings is needed (not included in the repository), e.g. [fastText embeddings](https://fasttext.cc/docs/en/crawl-vectors.html). The path to the embedding file can be given with the option `--embedding_file`. The default value for this option is 'data/balanced/embs/tokens.txt'.

The evaluation can be conducted with:

```
python evaluation.py
```

The option `--data_set` allows the three values 'train', 'val' and 'test' (with 'val' as default).

2. **Bilstm-mlp-elmo**

The first three steps are the same as for bilstm-mlp. To build the ELMo data set run:

```
python build_elmo_dataset.py ----elmo_params_dir path/to/options/and/weights
```

This will take quite some time. The weights and options file used for the experiments described in the paper can be found [here](https://github.com/t-systems-on-site-services-gmbh/german-elmo-model). The output will be again a train/val/test split, but this time tokens are followed by their respective ELMo embeddings. This is because, since we are using ELMo the same word has different embeddings in different sentences.

To build the average of the three different representations given by ELMo (for more information on this check out the [original paper](https://arxiv.org/pdf/1802.05365.pdf) run:

```
python compute_elmo_average.py
```

After that you can train and evaluate the model the same way as for bilstm-mlp, but you of course won't need to provide an embedding file for `train.py`.

**Note**: Every script has default values for command line options. If you want to use other values they have to be provided.


