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

        # Loading labels and map them to their indices
        labels_path = os.path.join(data_dir, 'labels_vocab.txt')
        self.labels_map = {}
        with open(labels_path, 'r', encoding='utf-8') as f:
            for i, label in enumerate(f.read().splitlines()):
                self.labels_map[label] = i

        self.inverse_labels_map = {i: label for (label, i) in self.labels_map.items()}

        # Adding dataset parameters to param
        params.update(json_path)

    def load_elmo_data(self, types, data_dir):
        data = {}
        
        for split in ['train', 'val', 'test']:
            if split in types:
                sentences_file = os.path.join(data_dir, split, 'sentences.txt')
                labels_file = os.path.join(data_dir, split, 'noun_verb_labels.txt')

                f = open(sentences_file, 'r', encoding='utf-8')

                print("Loading dataset into memory...")
                data_str = f.read()
                print("Done.")
                
                f.close()

                # Every sentence is representend by a list of np.arrays
                print("Extracting embedding layers...")
                sents = [[np.asarray(tok.split(' ')[1:], dtype='float32')
                        for tok in sent.split('\n')]
                        for sent in data_str.strip().split('\n\n')]
                print("Done.")

                labels = []

                with open(labels_file, 'r', encoding='utf-8') as f:
                    for tags in f.read().splitlines():
                        # replace each label by its index
                        l = [self.labels_map[tag] for tag in tags.split(' ')]
                        labels.append(l) 

                data[split] = {}
                data[split]['data'] = sents
                data[split]['labels'] = labels
                data[split]['size'] = len(sents)

        return data

    def elmo_iterator(self, data, params, shuffle=False):

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(42)
            random.shuffle(order)

        num_dims = len(data['data'][0][0])

        # one pass over data
        for i in range((data['size']+1)//params.batch_size):
            # fetch features and tags
            batch_sentences = [data['data'][idx] for idx
                              in order[i*params.batch_size:(i+1)*params.batch_size]]
            batch_tags = [data['labels'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]

            # compute length of longest sentence in batch
            batch_max_len = max([len(s) for s in batch_sentences])

            batch_data = np.zeros((len(batch_sentences), batch_max_len, num_dims))
            batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))

            # copy the data to the numpy array
            for i, sent in enumerate(batch_sentences):
                for j, token_embedding in enumerate(sent):
                    batch_data[i][j] = token_embedding
                    batch_labels[i][j] = batch_tags[i][j]

            batch_data, batch_labels = torch.tensor(batch_data, dtype=torch.float32, requires_grad=False),\
                                        torch.tensor(batch_labels, dtype=torch.float32, requires_grad=False)



            yield batch_data, batch_labels
