import argparse
from os import makedirs
from os.path import join, exists

from allennlp.commands.elmo import ElmoEmbedder

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/balanced',
help='Directory containing the dataset')
parser.add_argument('--elmo_params_dir',
help='Directory containing the options and weights for the allennlp ElmoEmbedder')

def load_dataset(splits, path_dataset, elmo_embedder):
    for split in splits:
        with open(join(path_dataset, split + '/tokens.txt'), 'r', encoding='utf-8') as f,\
             open(join(path_dataset, split + '/labels.txt'), 'r', encoding='utf-8') as l:

            sentences = [sent.strip().split() for sent in f.readlines()]
            labels = [tags.strip().split() for tags in l.readlines()]

            data = []
            sentence = []
            token_embeddings = []

            for i, sent in enumerate(sentences):
                assert len(sent) == len(labels[i])
                sentence_embeddings = elmo_embedder.embed_sentence(sent)
                for idx, token in enumerate(sent):
                    elmo_layers = fetch_token_embedding(sentence_embeddings, idx)
                    token_embeddings.append(token)
                    token_embeddings.append(elmo_layers)
                    sentence.append(token_embeddings)
                    token_embeddings = []
                data.append(sentence)
                sentence = []
                print(i)
            save_dataset(split, data)  

def save_dataset(split, data):
    if not exists(join('data/elmo', split)):
        makedirs(join('data/elmo', split))

    with open(join('data/elmo', split, 'sentences.txt'), 'w', encoding='utf-8') as e:
        for sent in data:
            for token, embedding in sent:
                e.write(token + ' ' + embedding + '\n')
            e.write('\n')

def fetch_token_embedding(sentence_embeddings, idx):
    token_embeddings = sentence_embeddings[:, idx]
    token_embeddings = token_embeddings.flatten()
    token_embeddings = ' '.join(map(str, token_embeddings))
    return token_embeddings

if __name__ == '__main__':
    args = parser.parse_args()

    path_dataset = args.data_dir
    elmo_params_dir = args.elmo_params_dir

    elmo_embedder = ElmoEmbedder(options_file=join(elmo_params_dir, 'options.json'),
                                 weight_file=join(elmo_params_dir, 'weights.hdf5'))

    load_dataset(['train', 'val', 'test'], path_dataset, elmo_embedder)