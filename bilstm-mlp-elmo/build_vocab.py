"""Builds vocabularies of tokens, lemmas, POS tags and labels"""

# Adapted from https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/build_vocab.py
# accessed on 03/09/2020

import argparse
import os
import json
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/balanced',
help="Directory containing the dataset")

PAD_WORD = '<pad>'
UNK_WORD = 'UNK'

def save_vocab_to_txt_file(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to the id of the token.

    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(token + '\n')
            
def save_dict_to_json(d, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w', encoding='utf-8') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

def update_vocab(txt_path, vocab):
    """Update the different vocabularies from the dataset

    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method

    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            vocab.update(line.strip().split(' '))

    return i + 1

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Build token vocab with train, val and test datasets
    print("Building token vocabulary...")
    tokens = Counter()
    size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train/tokens.txt'), tokens)
    size_dev_sentences = update_vocab(os.path.join(args.data_dir, 'val/tokens.txt'), tokens)
    size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test/tokens.txt'), tokens)
    print("Done.")

    # Build lemma vocab with train, val and test datasets
    print("Building lemma vocabulary...")
    lemmas = Counter()
    size_train_lemmas = update_vocab(os.path.join(args.data_dir, 'train/lemmas.txt'), lemmas)
    size_dev_lemmas = update_vocab(os.path.join(args.data_dir, 'val/lemmas.txt'), lemmas)
    size_test_lemmas = update_vocab(os.path.join(args.data_dir, 'test/lemmas.txt'), lemmas)
    print("Done.")

    # Build pos vocab with train, val and test datasets
    print("Building pos vocabulary...")
    pos = Counter()
    size_train_pos = update_vocab(os.path.join(args.data_dir, 'train/pos.txt'), pos)
    size_dev_pos = update_vocab(os.path.join(args.data_dir, 'val/pos.txt'), pos)
    size_test_pos = update_vocab(os.path.join(args.data_dir, 'test/pos.txt'), pos)
    print("Done.")

    # Build tag vocab with train and test datasets
    print("Building tag vocabulary...")
    tags = Counter()
    size_train_tags = update_vocab(os.path.join(args.data_dir, 'train/labels.txt'), tags)
    size_dev_tags = update_vocab(os.path.join(args.data_dir, 'val/labels.txt'), tags)
    size_test_tags = update_vocab(os.path.join(args.data_dir, 'test/labels.txt'), tags)
    print("Done.")

    # Assert same number of examples in datasets
    assert size_train_sentences == size_train_lemmas == size_train_pos == size_train_tags
    assert size_dev_sentences == size_dev_lemmas == size_dev_pos == size_dev_tags
    assert size_test_sentences == size_test_lemmas == size_test_pos == size_test_tags

    # Transorming to lists
    tokens = [tok for tok, count in tokens.items()]
    lemmas = [tok for tok, count in lemmas.items()]
    pos = [tok for tok, count in pos.items()]
    tags = [tok for tok, count in tags.items()]

    # Add pad tokens
    if PAD_WORD not in tokens: tokens.insert(0, PAD_WORD)
    if PAD_WORD not in lemmas: lemmas.insert(0, PAD_WORD)
    if PAD_WORD not in pos: pos.insert(0, PAD_WORD)
    
    # add word for unknown words
    tokens.insert(0, UNK_WORD)
    lemmas.insert(0, UNK_WORD)
    pos.insert(0, UNK_WORD)

    # Save vocabularies to file
    print("Saving vocabularies to file...")
    save_vocab_to_txt_file(tokens, os.path.join(args.data_dir, 'tokens_vocab.txt'))
    save_vocab_to_txt_file(lemmas, os.path.join(args.data_dir, 'lemmas_vocab.txt'))
    save_vocab_to_txt_file(pos, os.path.join(args.data_dir, 'pos_vocab.txt'))
    save_vocab_to_txt_file(tags, os.path.join(args.data_dir, 'labels_vocab.txt'))
    print("- done.")

    # Save datasets properties in json file
    sizes = {
        'train_size': size_train_sentences,
        'dev_size': size_dev_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(tokens),
        'number_of_lemmas': len(lemmas),
        'number_of_pos': len(pos),
        'number_of_labels': len(tags),
        'pad_word': PAD_WORD,
        'unk_word': UNK_WORD
    }
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))

    # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))