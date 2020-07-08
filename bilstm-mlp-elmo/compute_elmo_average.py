import argparse
import shutil
import numpy as np
from os import makedirs
from os.path import join, exists

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/elmo',
help='Directory containing the dataset')
parser.add_argument('--output_dir', default='data/averaged_elmo',
help='Output directory')
parser.add_argument('--original_data_dir', default='data/balanced',
help='Directory of the original data (before conversion to ELMo suitable foramt')

def convert_dataset(data_dir, output_dir, original_data_dir):
    for split in ['train', 'val', 'test']:
        if not exists(join(output_dir, split)):
            makedirs(join(output_dir, split))

        with open(join(data_dir, split, 'sentences.txt'), 'r', encoding='utf-8') as f,\
             open(join(output_dir, split, 'sentences.txt'), 'w', encoding='utf-8') as e:
             for line in f:
                if line not in ['\n']:
                    line = line.split()
                    token = line[0]
                    embedding_layers = line[1:]
                    averaged_layers = compute_layer_average(embedding_layers)
                    e.write(token + ' ' + ' '.join(map(str, averaged_layers)) + '\n')
                else:
                    e.write('\n')
        
        shutil.copyfile(join(original_data_dir, split, 'noun_verb_labels.txt'),
                        join(output_dir, split, 'noun_verb_labels.txt'))

        shutil.copyfile(join(original_data_dir, split, 'dataset_params.json'),
                        join(output_dir, split, 'dataset_params.json'))

        shutil.copyfile(join(original_data_dir, split, 'labels_vocab.txt'),
                        join(output_dir, split, 'labels_vocab.txt'))

def compute_layer_average(elmo_layers):
    elmo_layers = np.asarray(elmo_layers, dtype='float32')
    averaged_layers = np.mean(elmo_layers.reshape(3,-1), axis=0)
    return averaged_layers

if __name__ == '__main__':
    args = parser.parse_args()

    path_dataset = args.data_dir
    output_dir = args.output_dir
    original_data_dir = args.original_data_dir
    convert_dataset(path_dataset, output_dir, original_data_dir)