# Adapted from https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/utils.py
# accessed on 03/09/2020

import json
import logging
import os
import shutil

import gensim
import torch
import numpy as np

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

def load_w2v_embeddings(feature_map, model_path):
    """Loads pretrained word2vec embeddings from disk.

    Args:
        feature_map: (dict) Token or lemma map
        embedding_path: (string) Path to pretrained embedding model

    Returns:
        pretrained_weights: (numpy array)
    """
    decow_model = gensim.models.Word2Vec.load(model_path)
    vocab_size = len(feature_map)
    emb_dim = 100
    
    pretrained_weights = np.zeros((vocab_size, emb_dim))
    unknown_words = 0
    for word in feature_map:
        try:
            pretrained_weights[feature_map[word]] = decow_model[word]
        except KeyError as e:
            pretrained_weights[feature_map[word]] = np.random.normal(size=(emb_dim, ))
            unknown_words += 1

    return pretrained_weights

def load_embeddings_from_txt(feature_map, emb_path, embedding_dim=300):
    """Loads pretrained fasttext embeddings from disk.

    Args:
        feature_map: (dict) Feature map
        emb_path: (string) Path to embedding file
    """
    vocab_size = len(feature_map)
    emb_dim = embedding_dim
    pretrained_weights = np.zeros((vocab_size, emb_dim))

    with open(emb_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            emb = np.asarray(line.split()[1:], dtype='float32')
            pretrained_weights[idx] = emb

    # Change embeddings for unknown words and padding to random embeddings
    pretrained_weights[0] = np.random.normal(size=(emb_dim, ))
    pretrained_weights[1] = np.random.normal(size=(emb_dim, ))

    return pretrained_weights

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)