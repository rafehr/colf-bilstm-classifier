"""Reads the COLF-VID corpus and creates a balanced train/val/test split."""

import argparse
import os
import random
import json

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--corpus_dir', default='data/COLF-VID_1.0',
help='Directory of the COLF-VID corpus')
arg_parser.add_argument('--save_dir', default='data/balanced',
help='Output directory')
arg_parser.add_argument('--out_dir', default='data/balanced',
help='Output directory for the data set')
arg_parser.add_argument('--tokens', type=int, default=1,
help='Number of column tokens appear in')
arg_parser.add_argument('--lemmas', type=int, default=2,
help='Number of column lemmas appear in')
arg_parser.add_argument('--pos', type=int, default=3,
help='Number of column POS tags appear in')
arg_parser.add_argument('--heads', type=int, default=None,
help='Number of column heads indices appear in')
arg_parser.add_argument('--labels', type=int, default=7,
help='Number of column labels appear in')

def load_dataset(path_dataset):
    """Reads the COLF files.

    Args:
        path_dataset (string): Path to COLF dataset.

    Returns:
        sents_split_per_file (list): List of lists where every list contains the
            split sentences of the respective VID files in the original format
        vids (list): Contains the vid types
    """
    # Get the file path for every file in the COLF directory
    file_paths = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset)]
    
    # Create list of VIDs
    vids = [os.path.basename(file_path).strip('.txt') for file_path in file_paths]

    # Read the whole COLF dataset into one string and split it into sentences
    colf_str = ''
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as colf:
            colf_str += colf.read() + 'END_OF_FILE'

    sents_split_per_file = [f.strip().split('\n\n')
                            for f in colf_str.split('END_OF_FILE')[:-1]]

    return sents_split_per_file, vids

def split_dataset(sents_split_per_file, vids, feature_columns):
    """Splits the data in a balanced way.

    Splits the sentence ids into train, val and test by using the same ratios of
    sentences for every file. This is done because the different files contain
    different amounts of sentences.
    
    Args:
        sents_split_per_file (list): List of lists where every list contains the
            split sentences of the respective VID files in the original format
        vids (list): Contains the vid types
        feature_columns (dict): Contains features and their respective columns

    Returns:
        dataset: (list) List of lists. Every list containing tokens, lemmas,
            POS tags and tags.
        split_ids (dict): The sentence ids split into train, val and test.
        num_instances_per_vid (list): Contains the number of instances per VID
            for the val and test set.
    """
    # Create list of ids for every file
    ids_per_file = []
    range_begin, range_end = 0, 0

    for sents in sents_split_per_file:
        range_end += len(sents)
        ids_per_file.append([i for i in range(range_begin, range_end)])
        range_begin += len(sents)

    # Shuffling ids
    random.seed(13)
    for ids in ids_per_file:
        random.shuffle(ids)

    # Splitting the ids into train, val and test
    split_ids = {'train': [], 'val': [], 'test': []}
    num_instances_per_vid = []

    for ids, vid in zip(ids_per_file, vids):
        train_ids = ids[:int(0.7 * len(ids))]
        val_ids = ids[int(0.7 * len(ids)) : int(0.85 * len(ids))]
        test_ids = ids[int(0.85 * len(ids)):]

        split_ids['train'].extend(train_ids)
        split_ids['val'].extend(val_ids)
        split_ids['test'].extend(test_ids)

        num_instances_per_vid.append({vid: {'val': len(val_ids),
                                        'test': len(test_ids)}})

    dataset = []                              
    features_labels = {}
    for feature in feature_columns:
        features_labels[feature] = []

    # Extract features from COLF-VID files and construct dataset
    for sent in [sent.split('\n') for f in sents_split_per_file for sent in f]:
        for line in sent:
            if not line.startswith('#'):
                for feature in feature_columns:
                    column = feature_columns[feature]
                    split_line = line.split('\t')
                    features_labels[feature].append(split_line[column])
        dataset.append([features_labels[feature] for feature in features_labels])

        feature_lengths = []
        for feature in features_labels:
            feature_lengths.append(len(features_labels[feature]))

        assert len(set(feature_lengths)) == 1

        for feature in features_labels:
            features_labels[feature] = []

    return dataset, split_ids, num_instances_per_vid

def save_dataset(dataset, save_dir, feature_names):
    """Writes the features and labels to separate txt files in save_dir.
    
    Args:
        dataset (list): List of lists. Every list contains lists with the
            different features.
        save_dir (str): The output directory.
        feature_names (list): Names of the features.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    feature_files = []

    for f_name in feature_names:
        f_name = f_name + '.txt'
        feature_files.append(open(os.path.join(save_dir, f_name), 'w', encoding='utf-8'))

    for instance in dataset:
        for feature, ff in zip(instance, feature_files):
            ff.write("{}\n".format(" ".join(feature)))

    for ff in feature_files:
        ff.close()

if __name__ == '__main__':
    args = arg_parser.parse_args()
    colf_path = args.corpus_dir
    save_dir = args.save_dir

    feature_columns = {}

    for arg in vars(args):
        value = getattr(args, arg)
        if isinstance(value, int):
            feature_columns[arg] = value

    feature_names = list(feature_columns.keys())
    
    sents_split_per_file, vids = load_dataset(colf_path)
    dataset, split_ids, num_instances_per_vid = split_dataset(sents_split_per_file, vids, feature_columns)

    # Split the dataset into train, val and split
    train_dataset = [dataset[idx] for idx in split_ids['train']]
    val_dataset = [dataset[idx] for idx in split_ids['val']]
    test_dataset = [dataset[idx] for idx in split_ids['test']]

    # Save datasets
    save_dataset(train_dataset, os.path.join(save_dir, 'train'), feature_names)
    save_dataset(val_dataset, os.path.join(save_dir, 'val'), feature_names)
    save_dataset(test_dataset, os.path.join(save_dir, 'test'), feature_names)

     # Save number of instances per VID to JSON file
    with open(os.path.join(save_dir, 'num_instances_per_vid.json'), 'w', encoding='utf-8') as f:
        json.dump(num_instances_per_vid, f, ensure_ascii=False)



    