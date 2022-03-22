"""
component_mean.py

This program takes the average surprisal and reading time for each component
in an id file and saves them in a .tsv format.
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


def average_rts(rts_folder: str):
    """Go through all files in rts_folder and average the RT per token.
    
    Returns a pandas.Dataframe object with columns ["item", "zone", "RT"].
    """
    # cols = ["WorkerId", "WorkTimeInSeconds", "correct", "item", "zone", "RT"]
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
    word = ""
    tree, tree_start, tree_end = None, None, None
    item, zone_start, zone_end = None, None, None
    for line in lines:
        _, comp, ti, tw, ki, kz = line.split()
        if tree is None:
            tree, tree_start, tree_end = ti, tw, tw
            item, zone_start, zone_end = ki, kz, kz
        if (tree == ti and tree_end == tw) or \
            (item == ki and zone_end == kz):
            word += comp
            tree_end = tw
            zone_end = kz
        else:
            yield word, tree, tree_start, tree_end, item, zone_start, zone_end
            word = comp
            tree, tree_start, tree_end = ti, tw, tw
            item, zone_start, zone_end = ki, kz, kz


def main(args):
    """Take the average surprisal and reading time for each component."""
    surps = pd.read_csv(args.surp_file, sep='\t', header=0)
    rts = average_rts(args.rts_folder)
    lines = []

    for w, t, ts, te, z, zs, ze in get_full_tokens(args.id_file):

        cond = f"sent == {t} and token_id >= {ts} and token_id <= {te}"
        surp = surps.query(cond, inplace=False).sum()
        ind, brn = surp["ind"], surp["brn"]
        
        cond = f"item == {z} and zone >= {zs} and zone <= {ze}"
        rt = rts.query(cond, inplace=False).sum()["RT"]

        lines.append([len(lines), w, rt, ind, brn])

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
