"""Removes the VID labels for preposition"""

import argparse
import os

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data_dir', default='data/balanced/',
help='Directory of the dataset')

if __name__ == '__main__':
    args = arg_parser.parse_args()
    data_dir = args.data_dir

    pos_path_train = os.path.join(data_dir, 'train/pos.txt')
    pos_path_val = os.path.join(data_dir, 'val/pos.txt')
    pos_path_test = os.path.join(data_dir, 'test/pos.txt')

    labels_path_train = os.path.join(data_dir, 'train/labels.txt')
    labels_path_val = os.path.join(data_dir, 'val/labels.txt')
    labels_path_test = os.path.join(data_dir, 'test/labels.txt')

    with open(pos_path_train, 'r', encoding='utf-8') as p,\
        open(labels_path_train, 'r', encoding='utf-8') as l,\
        open(os.path.join(data_dir, 'train/noun_verb_labels.txt'), 'w', encoding='utf-8') as nvl:
        for pos, labels in zip(p, l):
            noun_verb_labels = []
            for p, l in zip(pos.split(), labels.split()):
                if p in ['APPRART', 'APPR'] and l != '*':
                    noun_verb_labels.append('*')
                else:
                    noun_verb_labels.append(l)
            assert len(labels.split()) == len(noun_verb_labels)
            assert len(noun_verb_labels) - noun_verb_labels.count('*') == 2
            nvl.write(' '.join(noun_verb_labels) + '\n')

    with open(pos_path_val, 'r', encoding='utf-8') as p,\
        open(labels_path_val, 'r', encoding='utf-8') as l,\
        open(os.path.join(data_dir, 'val/noun_verb_labels.txt'), 'w', encoding='utf-8') as nvl:
        for pos, labels in zip(p, l):
            noun_verb_labels = []
            for p, l in zip(pos.split(), labels.split()):
                if p in ['APPRART', 'APPR'] and l != '*':
                    noun_verb_labels.append('*')
                else:
                    noun_verb_labels.append(l)
            assert len(labels.split()) == len(noun_verb_labels)
            assert len(noun_verb_labels) - noun_verb_labels.count('*') == 2
            nvl.write(' '.join(noun_verb_labels) + '\n')

    with open(pos_path_test, 'r', encoding='utf-8') as p,\
        open(labels_path_test, 'r', encoding='utf-8') as l,\
        open(os.path.join(data_dir, 'test/noun_verb_labels.txt'), 'w', encoding='utf-8') as nvl:
        for pos, labels in zip(p, l):
            noun_verb_labels = []
            for p, l in zip(pos.split(), labels.split()):
                if p in ['APPRART', 'APPR'] and l != '*':
                    noun_verb_labels.append('*')
                else:
                    noun_verb_labels.append(l)
            assert len(labels.split()) == len(noun_verb_labels)
            assert len(noun_verb_labels) - noun_verb_labels.count('*') == 2
            nvl.write(' '.join(noun_verb_labels) + '\n')