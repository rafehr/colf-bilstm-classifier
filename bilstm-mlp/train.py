import utils
import lstm as net

import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import data_loader

from sklearn.metrics import accuracy_score

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--embedding_file', default='data/balanced/embs/tokens.txt',
help='Directory containing file with embeddings')
arg_parser.add_argument('--model_name', default='colf_classifier.pt',
help='Name of the saved model')
arg_parser.add_argument('--data_dir', default='data/balanced',
help='Directory containing the data and dataset parameters')
arg_parser.add_argument('--param_dir', default='experiments/base_model',
help='Directory containing hyperparameters')

if __name__ == '__main__':
    args = arg_parser.parse_args()
    path_embedding_file = args.embedding_file
    model_name = args.model_name
    data_dir = args.data_dir
    params_dir = args.param_dir

    params = utils.Params(os.path.join(data_dir,'dataset_params.json'))
    params.update(os.path.join(params_dir, 'params.json'))

    dl = data_loader.DataLoader('data/balanced/', params)
    data = dl.load_data(['train'], 'data/balanced')

    pretrained_weights = utils.load_embeddings_from_txt(dl.feature_map,
                            path_embedding_file)

    net = net.Network(params, pretrained_weights)

    learning_rate = params.learning_rate

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(params.num_epochs):

        data_iter = dl.data_iterator(data['train'], params, shuffle=True)

        total_loss = 0
        total_correct = 0

        # Training loop
        for batch in data_iter:
            sents, labels = batch

            preds, single_labels = net(sents, labels)
            loss = F.cross_entropy(preds, single_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += net.get_num_correct(preds, single_labels)

        print("epoch: ", epoch + 1, "total_correct: ", total_correct, "total_loss: ", total_loss, flush=True)

    torch.save(net.state_dict(), model_name)