import utils
import lstm as net

import torch
import torch.nn.functional as F
import torch.optim as optim
import data_loader

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

if __name__ == '__main__':
    params = utils.Params('data/balanced/dataset_params.json')
    params.update('experiments/elmo_model/params.json')

    dl = data_loader.DataLoader('data/averaged_elmo/', params)
    data = dl.load_elmo_data(['train'], 'data/averaged_elmo')

    net = net.Network(params)

    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    for epoch in range(params.num_epochs):

        data_iter = dl.elmo_iterator(data['train'], params, shuffle=True)

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

    torch.save(net.state_dict(), 'bilstm_mlp_elmo.pt')
