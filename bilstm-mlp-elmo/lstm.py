import numpy as np
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data_loader

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

class Network(nn.Module):
    def __init__(self, params):
        super(Network, self).__init__()

        self.params = params

        self.lstm = nn.LSTM(self.params.pretrained_embedding_dim, self.params.lstm_hidden_dim,
                            batch_first=True, bidirectional=True)

        self.fc = nn.Linear(in_features=(self.params.lstm_hidden_dim * 4), out_features=100)
        self.out = nn.Linear(in_features=100, out_features=5)

    def forward(self, t, labels):
        # Get the indices of the tokes that are annotated in a sentence
        idx_list = []
        
        for l in labels:
            idxs = [idx for idx, tag in enumerate(l) if tag not in [0, -1]]
            assert len(idxs) == 2
            idx_list.append(idxs)
        
        # Create labels tensor with one label per sentence (i.e. one label per MWE token)
        single_labels = [l[idxs[0]] for l, idxs in zip(labels, idx_list)]
        single_labels = torch.tensor(single_labels)
        single_labels = single_labels.long()

        context_embs, _ = self.lstm(t)

        concat_embs = torch.empty((self.params.batch_size, self.params.lstm_hidden_dim * 4))

        for j in range(self.params.batch_size):
            concat_embs[j] = torch.cat((context_embs[j, idx_list[j][0]], context_embs[j, idx_list[j][1]]))

        t = self.fc(concat_embs)
        t = torch.tanh(t)
        t = self.out(t)

        return t, single_labels

    @staticmethod
    def get_num_correct(preds, labels):
        return torch.argmax(preds, dim=1).eq(labels).sum().item()