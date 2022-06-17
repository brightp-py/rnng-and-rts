"""
evaluate_all_words.py.

Based on evaluate_target_word.py.

But like... I want all the words, please.

Brighton Pauli
"""

import argparse
from tqdm import tqdm

import torch
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

BATCH_SIZE = 1
SEQ_LEN = 20


def evaluate(model, data_source, trees_source, tokens_source, dictionary):
    model.eval()

    hidden = model.init_hidden(BATCH_SIZE)

    with torch.no_grad():
        for ind in range(0, data_source.size(0) - 1, SEQ_LEN):
            data, targets = get_batch(data_source, ind, SEQ_LEN)
            _, trees = get_batch(trees_source, ind, SEQ_LEN)
            _, tokens = get_batch(tokens_source, ind, SEQ_LEN)
            output, hidden = model(data, hidden)
            output = output.view(-1, len(dictionary))
            hidden = repackage_hidden(hidden)

            log_probs_np = -F.log_softmax(output, dim=1).cpu().numpy()
            targets = targets.cpu().numpy()
            trees = trees.cpu().numpy()
            tokens = tokens.cpu().numpy()

            data_lens = [data.shape[0] for data in
                         (log_probs_np, targets, trees, tokens)]
            data_desc = ' | '.join(map(lambda x: str(x).rjust(2), data_lens))
            min_seq_len = min(data_lens)

            for tok_ind in range(min_seq_len):
                target = targets[tok_ind]
                score = str(float(log_probs_np[tok_ind][target]))
                word = dictionary.idx2word[target]
                tree, tok = str(trees[tok_ind]), str(tokens[tok_ind])
                if word != "<eos>":
                    yield f"{tree}\t{tok}\t{word}\t{score}", \
                        f"{tree.rjust(3)} {tok.rjust(3)} ( {data_desc} )"


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

    with open(args.checkpoint, 'rb') as file:
        print("Loading the model")
        if args.cuda:
            model = torch.load(file)
        else:
            model = torch.load(file, map_location=lambda storage, loc: storage)

    model.eval()

    if args.cuda:
        model.cuda()
    else:
        model.cpu()

    dictionary = dictionary_corpus.Dictionary(args.data)

    trees, tokens = create_id_mapping(args.path + ".text")
    tree_data = batchify(torch.LongTensor(trees), BATCH_SIZE, args.cuda)
    token_data = batchify(torch.LongTensor(tokens), BATCH_SIZE, args.cuda)

    test_data = batchify(
        dictionary_corpus.tokenize(dictionary, args.path + ".text"),
        BATCH_SIZE, args.cuda
    )

    with open(args.path + ".text", 'r', encoding='utf-8') as f_read:
        first_word = f_read.read().split()[0]

    with open(args.path + "_lstm.output", 'w', encoding='utf-8') as f_output:
        f_output.write(
            f"sent\tsent_pos\tword\tlstm_surp\n1\t0\t{first_word}\t0\n")

        pbar = tqdm(evaluate(model, test_data, tree_data, token_data,
                             dictionary))
        for row, desc in pbar:
            f_output.write(row + '\n')
            pbar.set_description(desc)


if __name__ == "__main__":
    arguments = parser.parse_args()
    main(arguments)
