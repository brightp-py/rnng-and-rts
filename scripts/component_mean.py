"""
component_mean.py.

** Currently poorly named! Actually saves median! **

This program takes the average surprisal and reading time for each component
in an id file and saves them in a .tsv format.

Brighton Pauli
"""

import os
import argparse

import pandas as pd

parser = argparse.ArgumentParser()

DATADIR = 'D:/data/naturalstories'

# Data path options
parser.add_argument('--id_file', default=f'{DATADIR}/ids.tsv')
parser.add_argument('--surp_file',
                    default=r'C:\git\rnng-and-rts\rnng\data\ns-results.tsv')
parser.add_argument('--rts_folder', default=f'{DATADIR}/rts-raw/')
parser.add_argument('--save_file', default=f'{DATADIR}/rt-surp.tsv')


class Token:
    """A token of text, as from the ids file.

    Attributes:
        * word - str of this token.
        * tree - Sentence index.
        * tree_start - Where this token starts in the tree.
        * tree_end - Where this token ends in the tree.
                     Optimally the same as tree_start.
        * item - Story index.
        * zone_start - Where this token starts in the story.
        * zone_end - Where this token ends in the story.
                     Optimally the same as zone_start.
    """

    def __init__(self, word, tree, tree_start, item, zone_start):
        """Start a new, possibly incomplete token.

        tree_end and zone_end are initialized as tree_start and zone_start.
        """
        self.word = word
        self.tree = tree
        self.tree_start = tree_start
        self.tree_end = tree_start
        self.item = item
        self.zone_start = zone_start
        self.zone_end = zone_start

    def add_component(self, component, tree_end, zone_end):
        """Add a component to this token."""
        self.word += component
        self.tree_end = tree_end
        self.zone_end = zone_end

    def tree_cond(self):
        """Set up a condition that finds this token in tree data."""
        cond = f"sent == {self.tree} and word_num >= {self.tree_start} " \
               f"and word_num <= {self.tree_end}"
        return cond

    def rt_cond(self):
        """Set up a condition that finds this token in reading time data."""
        cond = f"item == {self.item} and zone >= {self.zone_start} " \
               f"and zone <= {self.zone_end}"
        return cond


def average_rts(rts_folder: str):
    """Go through all files in rts_folder and average the RT per token.

    Returns a pandas.Dataframe object with columns ["item", "zone", "RT"].
    """
    data = None
    for root, _, files in os.walk(rts_folder):
        print(f"Found {int(len(files))} reading time files: {' '.join(files)}")
        for file in files:
            ndata = pd.read_csv(os.path.join(root, file), header=0)
            if data is None:
                data = ndata
            else:
                data = pd.concat((data, ndata))
    return data.groupby(by=["item", "zone"]).median() \
               .drop(columns=["WorkTimeInSeconds", "correct"])


def get_full_tokens(id_file: str):
    """Yield tokens of conjoined components, with tree and tok indices."""
    with open(id_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()[1:]
    cur_token = None
    for line in lines:
        _, comp, tree, tree_loc, item, zone_loc = line.split()
        if cur_token is None:
            cur_token = Token(comp, tree, tree_loc, item, zone_loc)
        if (tree == cur_token.tree and tree_loc == cur_token.tree_end) or \
           (item == cur_token.item and zone_loc == cur_token.zone_end):
            cur_token.add_component(comp, tree_loc, zone_loc)
        else:
            yield cur_token
            cur_token = Token(comp, tree, tree_loc, item, zone_loc)


def main(args):
    """Take the average surprisal and reading time for each component."""
    surps = pd.read_csv(args.surp_file, sep='\t', header=0)
    rts = average_rts(args.rts_folder)
    lines = []

    for token in get_full_tokens(args.id_file):
        surp = surps.query(token.tree_cond(), inplace=False).sum()
        ind, brn = surp["ind"], surp["brn"]
        read_time = rts.query(token.rt_cond(), inplace=False).sum()["RT"]
        lines.append([len(lines), token.word, read_time, ind, brn])
        if not len(lines) % 1000:
            print(len(lines))

    with open(args.save_file, 'w', encoding='utf-8') as file:
        file.write(
            "item\tword\treading-time\tindep-surp\tbrn-surp\n" +
            '\n'.join(map(
                lambda x: '\t'.join(map(str, x)),
                lines
            ))
        )

    print(f"Saved to {args.save_file}.")


if __name__ == "__main__":
    arguments = parser.parse_args()
    main(arguments)
