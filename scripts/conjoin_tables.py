"""
conjoin_tables.py.

Bring together two tables:
 - Reading times by subject by token
 - Surprisal by token

This should be the final step before R anaysis.

Ideally, this process would be included in the R analysis to lower the number
of steps needed to get data visualizations, but this Python script will fill
that role for now.

Brighton Pauli
"""

import os
import argparse
from functools import cache
from tqdm import tqdm

import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--rnng_file", default="analysis/data/ns-results.tsv")
parser.add_argument("--lstm_file")
parser.add_argument("--rt_dir", default="analysis/data/rts-raw")
parser.add_argument("--id_file", default="analysis/data/ids.tsv")
parser.add_argument("--save_file", default="analysis/data/final.csv")

def get_rts(rts_folder):
    data = None
    for root, _, files in os.walk(rts_folder):
        print(f"Found {int(len(files))} reading time files: {' '.join(files)}")
        for file in files:
            ndata = pd.read_csv(os.path.join(root, file), header=0)
            if data is None:
                data = ndata
            else:
                data = pd.concat((data, ndata))
    return data


def cut_malformed(surps: pd.DataFrame, ids: pd.DataFrame, drop_cols=None):
    if not drop_cols:
        drop_cols = []

    valid = []
    failed = []
    for surp, id in zip(surps.groupby(['sent']), ids.groupby(['TreeInd'])):
        if len(surp[1]) == len(id[1]):
            valid.append(pd.merge_ordered(surp[1], id[1],
                left_on="token_id", right_on="TreeWord")
                .drop(columns=['Index', 'Component', 'TreeInd', 'TreeWord']))
        else:
            failed.append(surp[0])
    if failed:
        print(f"The following {str(len(failed))} sentences failed:")
        print(' '.join(map(str, failed)))
    data = pd.concat(valid)

    @cache
    def f(item, zone, colname):
        return data.loc[
            (data['TokenItem']==item) & (data['TokenZone']==zone)] \
            .sum()[colname] # .drop(columns=drop_cols)

    return f
    

def main(args):
    ids = pd.read_csv(args.id_file, sep='\t')
    rts = get_rts(args.rt_dir).sort_values('zone')

    rnng = pd.read_csv(args.rnng_file, sep='\t')
    lstm = pd.read_csv(args.lstm_file, sep='\t')

    print("Gathering RNNG data.")
    get_surp = cut_malformed(rnng, ids, drop_cols=['item', 'sent', 'token_id',
        'depth', 'TokenItem', 'TokenZone', 'word'])
    
    rts['ind'] = rts.apply(lambda row: get_surp(row['item'], row['zone'],
        'ind'), axis=1)
    
    print("Gathering LSTM data.")
    get_surp = cut_malformed(lstm, ids)
    
    rts['lstm'] = rts.apply(lambda row: get_surp(row['item'], row['zone'],
        'surp'), axis=1)
    
    print(len(rts), "reading times found.")
    print(f"Saving to {args.save_file}.")

    rts.sort_values('WorkerId').to_csv(args.save_file)

if __name__ == "__main__":
    arguments = parser.parse_args()
    main(arguments)
