import lstm as net
import utils
import json
import argparse

import torch
import data_loader

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data_set', default='val', choices=['train', 'val', 'test'],
help='The data set you want to evaluate')
arg_parser.add_argument('--model', default='bilstm_mlp_elmo.pt', help='Model name')


if __name__ == '__main__':
    args = arg_parser.parse_args()
    data_set = args.data_set
    model = args.model

    params = utils.Params('data/balanced/dataset_params.json')
    params.update('experiments/elmo_model/params.json')

    dl = data_loader.DataLoader('data/averaged_elmo/', params)
    data = dl.load_elmo_data([data_set], 'data/averaged_elmo')

    net = net.Network(params)

    net.load_state_dict(torch.load(model))

    # Evaluation
    val_data_iter = dl.elmo_iterator(data[data_set], params, shuffle=False)

    total_correct = 0

    predictions = torch.tensor([], dtype=torch.long)
    true_labels = torch.tensor([], dtype=torch.long)

    for batch in val_data_iter:
        sents, labels = batch

        preds, single_labels = net(sents, labels)
        total_correct += net.get_num_correct(preds, single_labels)
        preds = torch.argmax(preds, dim=1)

        predictions = torch.cat((predictions, preds),dim=0)
        true_labels = torch.cat((true_labels, single_labels), dim=0)

    num_instances = len(predictions)

    print("#### EVALUATION RESULTS ####")
    print("Number of instances: ", num_instances)
    print("total_correct: ", total_correct)

    print(confusion_matrix(true_labels, predictions))
    print(precision_recall_fscore_support(true_labels, predictions, average='weighted'))