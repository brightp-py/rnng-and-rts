#!/usr/bin/env python3
"""Transform the json file from evaluate_trees into a tsv file.

Grabs the surprisal at a certain point of each syntax tree and saves the
result in a tsv file.

Currently, this program saves two types of surprisals for each word in each
sentence. The first is simply the surprisal of that word given the syntactic
context. The second is the sum of the surprisals of the word AND each of its
parent nonterminals.

The tsv's columns are:
    * The classifier for the sentence (caus.act/cos.tran/intr)
    * The numeric identifer for the sentence
    * The id of the word in that sentence
    * The word as the RNNG model sees it
    * The first, naive type of surprisal
    * The second, additive type of suprisal

The program also needs to grab these ids from a separate file. It expects the
first three columns of this id file to be:
    * The classifier for the sentence (caus.act/cos.tran/intr)
    * The numeric identifer for the sentence
    * The entire sentence
"""
import argparse
import json
import re
from collections import deque

parser = argparse.ArgumentParser()

parser.add_argument('--file', default='data/ns-results.json')
parser.add_argument('--save_file', default='data/ns-results.tsv')
parser.add_argument('--id_file', default='D:/data/naturalstories/ids.csv')

is_nt = re.compile(r'NT\(.*\)')


def grab_ids(file_name):
    """Extract sentence ids from a csv.

    The file that `file_name` points to should be a csv of three columns,
    where the first is the three-word identifier, the second is the sentence's
    number, and the third is the sentence itself.

    Yields lists [name, number, sentence].
    """
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        yield line.split(',', maxsplit=2)


def word_surps(instructions, tokens):
    """Yield the surprisal of each terminal node of the given tree.
    
    Yields tuples (ID, WORD, IND, BRN), where
        ID is the index of the word in this sentence
        WORD is the word itself, as a string
        IND is the raw surprisal of the word
        BRN is the added surprisal of the word with all its parent NTs.
    """
    total = 0
    stack = deque()
    i = 0
    for inst, surp in instructions:
        if inst == "REDUCE":
            continue
        total += surp
        stack.append(surp)
        if inst == "SHIFT":
            word_surp = tokens[i][1]
            yield i, tokens[i][0], word_surp, word_surp + total
            i += 1


def main(args):
    with open(args.file, 'r', encoding='utf-8') as file:
        results = json.load(file)

    lines = [
        "item\tsent\ttoken_id\tword\tind\tbrn"
    ]

    for i, (name, num, _) in enumerate(grab_ids(args.id_file)):
        actions = results[str(i)]['actions']
        tokens = results[str(i)]['tokens']
        for id, word, ind, brn in word_surps(actions, tokens):
            lines.append(
                f"{name}\t{num}\t{str(id)}\t{word}\t{str(ind)}\t{str(brn)}"
            )

    with open(args.save_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines))


if __name__ == "__main__":
    arguments = parser.parse_args()
    main(arguments)
