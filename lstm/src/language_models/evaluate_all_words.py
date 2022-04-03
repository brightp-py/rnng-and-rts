"""
evaluate_all_words.py.

Based on evaluate_target_word.py.

But like... I want all the words, please.

Brighton Pauli
"""

import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import dictionary_corpus
from utils import repackage_hidden, batchify, get_batch
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str,
    help='location of the data corpus for LM training')
parser.add_argument('--checkpoint', type=str,
    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
    help='random seed')
parser.add_argument('--cuda', action='store_true',
    help='use CUDA')
parser.add_argument('--path', type=str,
    help='path to test file (text) gold file (indices of words to evaluate)')


def evaluate(model, data_source, trees, tokens, d, eval_batch_size, seq_len, vocab_size):
    model.eval()
    
    hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            data, targets = get_batch(data_source, i, seq_len)
            _, trees = get_batch(trees, i, seq_len)
            _, tokens = get_batch(tokens, i, seq_len)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, vocab_size)
            hidden = repackage_hidden(hidden)

            # yield from output_candidates_probs(output_flat)
            log_probs_np = F.log_softmax(output_flat, dim=1).cpu().numpy()
            for scores, t, tree, tok in zip(log_probs_np, targets.cpu().numpy(),
            trees.cpu().numpy(), tokens.cpu().numpy()):
                yield f"{str(float(scores[t]))}\t{d.idx2word[t]}\t{str(tree)}\t{str(tok)}"


# def create_target_mask(test_file):
#     sents = open(test_file, "r").readlines()
#     len_s = sum(len(sent.split(" ")) for sent in sents)
#     targets = np.ones(len_s, dtype=float)
#     return targets


def create_id_mapping(test_file):
    sents = open(test_file, "r").readlines()
    trees = []
    tokens = []
    for i, sent in enumerate(sents):
        for j in range(len(sent.split(' '))):
            trees.append(i+1)
            tokens.append(j)
    return np.array(trees), np.array(tokens)


def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a Cuda device, so you should run --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    with open(args.checkpoint, 'rb') as f:
        print("Loading the model")
        if args.cuda:
            model = torch.load(f)
        else:
            model = torch.load(f, map_location=lambda storage, loc: storage)
    
    model.eval()

    if args.cuda:
        model.cuda()
    else:
        model.cpu()
    
    batch_size = 1000
    seq_len = 20

    dictionary = dictionary_corpus.Dictionary(args.data)
    vocab_size = len(dictionary)

    trees, tokens = create_id_mapping(args.path + ".text")
    tree_data = batchify(torch.LongTensor(trees), batch_size, args.cuda)
    token_data = batchify(torch.LongTensor(tokens), batch_size, args.cuda)

    test_data = batchify(
        dictionary_corpus.tokenize(dictionary, args.path + ".text"),
        batch_size, args.cuda
    )

    with open(args.path + ".output", 'w', encoding='utf-8') as f_output:
        for row in tqdm(
                evaluate(model, test_data, tree_data, token_data, dictionary,
                    batch_size, seq_len, vocab_size)):
            f_output.write(row + '\n')
            # pass


if __name__ == "__main__":
    arguments = parser.parse_args()
    main(arguments)
