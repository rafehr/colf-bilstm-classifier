import numpy as np
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data_loader

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

class Network(nn.Module):
    def __init__(self, params, pretrained_weights):
        super(Network, self).__init__()

        self.params = params

        self.embedding, self.emb_size = create_embedding_layer(pretrained_weights)

        self.lstm = nn.LSTM(self.emb_size, self.params.lstm_hidden_dim,
                            batch_first=True, bidirectional=True)

        self.fc = nn.Linear(in_features=(self.params.lstm_hidden_dim * 4), out_features=100)
        self.out = nn.Linear(in_features=100, out_features=5)

    def forward(self, t, labels):
        # Get the indices of the nound and verb tokens annotated as VIDs in a sentence
        idx_list = []
        
        for l in labels:
            idxs = [idx for idx, tag in enumerate(l) if tag not in [0, -1]]
            assert len(idxs) == 2
            idx_list.append(idxs)
        
        # Create labels tensor with one label per sentence (i.e. one label per VID instance)
        single_labels = [l[idxs[0]] for l, idxs in zip(labels, idx_list)]
        single_labels = torch.tensor(single_labels)
        single_labels = single_labels.long()

        t = t.clone().detach()
        t = t.long()
        t.requires_grad_(False)
        t = self.embedding(t)

        batch_size = t.shape[0]

        context_embs, _ = self.lstm(t)

        concat_embs = torch.empty((batch_size, self.params.lstm_hidden_dim * 4))

        for j in range(batch_size):
            concat_embs[j] = torch.cat((context_embs[j, idx_list[j][0]], context_embs[j, idx_list[j][1]]))

        t = self.fc(concat_embs)
        t = torch.tanh(t)
        t = self.out(t)

        return t, single_labels

    @staticmethod
    def get_num_correct(preds, labels):
        return torch.argmax(preds, dim=1).eq(labels).sum().item()

def create_embedding_layer(pretrained_weights):
    emb_size = pretrained_weights.shape[1]
    weight = torch.FloatTensor(pretrained_weights)
    embedding = nn.Embedding.from_pretrained(weight)
    return embedding, emb_size
