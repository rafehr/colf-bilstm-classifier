# Adapted from https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/model/data_loader.py
# accessed on 03/09/2020

import random
import numpy as np
import os
import sys

import torch

import utils

class DataLoader():
    """
    Handles all aspects of the data. Creates a feature map, batches the data and
    stores the dataset parameters.
    """
    def __init__(self, data_dir, params, tokens=True):
        """
        Loads dataset parameters, vocabulary and labels.

        Args:
            data_dir: (str) Directory containing the dataset.
            params: (Params) Hyperparameters of the training process.
        """
        self.tokens = tokens

        # Loading dataset parameters
        json_path = os.path.join(data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
        self.dataset_params = utils.Params(json_path)

        # Loading tokens or lemmas and map them to their indices
        if tokens:
            feature_path = os.path.join(data_dir, 'tokens_vocab.txt')
        else:
            feature_path = os.path.join(data_dir, 'lemmas_vocab.txt')
        self.feature_map = {}
        with open(feature_path, 'r', encoding='utf-8') as f:
            for i, feature in enumerate(f.read().splitlines()):
                self.feature_map[feature] = i

        self.inverse_feature_map = {i: feature for (feature, i) in self.feature_map.items()}

        self.unk_ind = self.feature_map[self.dataset_params.unk_word]
        self.pad_ind = self.feature_map[self.dataset_params.pad_word]

        # Loading labels and map them to their indices
        labels_path = os.path.join(data_dir, 'labels_vocab.txt')
        self.labels_map = {}
        with open(labels_path, 'r', encoding='utf-8') as f:
            for i, label in enumerate(f.read().splitlines()):
                self.labels_map[label] = i

        self.inverse_labels_map = {i: label for (label, i) in self.labels_map.items()}

        # Adding dataset parameters to param
        params.update(json_path)

    def load_features_labels(self, feature_path, labels_path, data):
        """
        Loads features and labels from their corresponding files, maps them
        to their indices and stores them in the data dict.

        Args:
            feature_path: (str) Path to feature file with features space-separated
            data: (dict) A dictionary in which the loaded data is stored
        """
        sentences, labels = [], []

        with open(feature_path, 'r', encoding='utf-8') as f:
            for line in f.read().splitlines():
                s = [self.feature_map[i] if i in self.feature_map
                     else self.unk_ind for i in line.split(' ')]
                sentences.append(s)

        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f.read().splitlines():
                l = [self.labels_map[t] for t in line.split(' ')]
                labels.append(l)

        # Checks to ensure there is a label for each token
        assert len(labels) == len(sentences)
        for l, t in zip(labels, sentences):
            assert len(l) == len(t)

        data['data'] = sentences
        data['labels'] = labels
        data['size'] = len(sentences)

    def load_data(self, types, data_dir):
        """
        Loads the data for each type in types from data_dir.

        Args:
            types: (list) has one or more of 'train', 'val', 'test' depending on 
            which data is required
            data_dir: (string) directory containing the dataset

        Returns:
            data: (dict) contains the data with labels for each type in types
        """
        data = {}
        
        for split in ['train', 'val', 'test']:
            if split in types:
                if self.tokens:
                    feature_path = os.path.join(data_dir, split, 'tokens.txt')
                else:
                    feature_path = os.path.join(data_dir, split, 'lemmas.txt')
                labels_path = os.path.join(data_dir, split, 'noun_verb_labels.txt')
                data[split] = {}
                self.load_features_labels(feature_path, labels_path, data[split])

        return data

    def data_iterator(self, data, params, shuffle=False):
        """
        Returns a generator that yields data batches with labels. Expires after
        one pass over data.

        Args:
            data: (dict) Contains data with keys 'data', 'labels' and 'size'
            params: (Params) Hyperparameters of the training process
            shuffle: (bool) Whether the data should be shuffled

        Yields:
            batch_data: (Tensor) A batch of sentences with dimensions batch_size x seq_len
            batch_labels: (Tensor) A batch of labels with dimensions batch_size x seq_len
        """
        # Make a list that decides the order in which we go over the data
        order = list(range(data['size']))
        if shuffle:
            random.seed(42)
            random.shuffle(order)

        # One pass over data
        for i in range((data['size']//params.batch_size) + 1):
            # Fetch features and tags
            batch_sentences = [data['data'][idx] for idx in
                               order[i*params.batch_size:(i+1)*params.batch_size]]
            batch_tags = [data['labels'][idx] for idx in
                            order[i*params.batch_size:(i+1)*params.batch_size]]

            # Compute length of longest sentence in batch
            batch_max_len = max([len(s) for s in batch_sentences])

            # Prepare a numpy array with the data, initialising the data with
            # pad_ind and all labels with -1 initialising labels to -1
            # differentiates tokens with tags from padding tokens
            batch_data = self.pad_ind * np.ones((len(batch_sentences), batch_max_len))
            batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))

            # Copy the data to the numpy array
            for j in range(len(batch_sentences)):
                cur_len = len(batch_sentences[j])
                batch_data[j][:cur_len] = batch_sentences[j]
                batch_labels[j][:cur_len] = batch_tags[j]

            batch_data, batch_labels = torch.tensor(batch_data, requires_grad=True),\
                                        torch.tensor(batch_labels, requires_grad=True)

            if params.cuda:
                batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
    
            yield batch_data, batch_labels
