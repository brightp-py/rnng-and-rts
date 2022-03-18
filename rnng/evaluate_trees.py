#!/usr/bin/env python3
"""Evaluate provided sentences as syntax trees.

Get surprisals for syntax trees using a provided RNNG. Provide pre-processed
data and a trained RNNG model.

This script uses PyTorch v1.9.1.

This file should not normally be imported as a module, but doing so would
provide the following functions:

    * load_model - returns a pytorch model from a loaded model file name
    * eval_ppl - returns a dictionary of perplexity values for given data
    * main - the main function of the script (args are required)
"""
import argparse
import json
import logging
import numpy as np

import torch

from data import Dataset
from train import create_model

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('--test_file', default='data/caus-test.json')
parser.add_argument('--model_file', default='rnng.pt')
parser.add_argument('--save_to', default='')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                    help='If "cuda", GPU number --gpu is used.')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--strategy', default='top_down',
                    choices=['top_down', 'in_order'])
parser.add_argument('--batch_token_size', type=int, default=15000,
                    help='Number of tokens in a batch (batch_size*'
                    'sentence_length) does not exceed this.')
parser.add_argument('--batch_action_size', type=int, default=45000,
                    help='(batch_size*max_action_length) does not exceed this.'
                    )
parser.add_argument('--batch_group',
                    choices=['same_length', 'random', 'similar_length',
                             'similar_action_length'],
                    default='similar_length',
                    help='Sentences are grouped by this criterion to make'
                    ' each batch.')
parser.add_argument('--max_group_length_diff', default=20, type=int,
                    help='When --batch_group=similar_length or similar_action'
                    '_length, maximum (token or action) length difference in'
                    ' a single batch does not exceed this.')
parser.add_argument('--group_sentence_size', default=1024, type=int,
                    help='When --batch_group=similar_length, sentences are'
                    ' first sorted by length and grouped by this number of'
                    ' sentences, from which each batch is sampled.')


def load_model(checkpoint, action_dict, vocab):
    """Create or retrieve a loaded model.

    All parameters can be retrieved from the torch.load function.
    """
    if 'model_state_dict' in checkpoint:
        model = create_model(checkpoint['args'], action_dict, vocab)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    return checkpoint['model']


def eval_ppl(data, model, vocab):
    """Create a dictionary of sentences' surprisals.

    A surprisal is assigned to every action and terminal token.
    """
    device = next(model.parameters()).device
    model.eval()

    losses = {}

    with torch.no_grad():
        for batch in data.batches():
            token_ids, action_ids, max_stack, sw_end_mask, batch_idx = batch

            token_ids = token_ids.to(device)
            action_ids = action_ids.to(device)
            sw_end_mask = sw_end_mask.to(device)

            _, a_loss, w_loss, _ = model(token_ids, action_ids,
                                         stack_size_bound=max_stack)

            start_a = 0
            start_w = 0

            for i, batch_id in enumerate(batch_idx):

                batch_id = int(batch_id)
                losses[batch_id] = {}

                last_action = action_ids[i].count_nonzero()
                losses[batch_id]['actions'] = [
                    (model.action_dict.i2a[a], float(loss)) for a, loss in zip(
                        action_ids[i][:last_action],
                        a_loss[start_a:start_a+last_action]
                    )
                ]
                start_a += last_action

                nshifts = (action_ids[i] == 1).count_nonzero()
                losses[batch_id]['tokens'] = [
                    (vocab.id_to_word(w), float(loss)) for w, loss in zip(
                        token_ids[i][:nshifts],
                        w_loss[start_w:start_w+nshifts]
                    )
                ]

    return losses


def main(args):

    logger.info('Args: %s', args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    checkpoint = torch.load(args.model_file)
    action_dict = checkpoint['action_dict']
    vocab = checkpoint['vocab']
    model = load_model(checkpoint, action_dict, vocab).to(device)

    if args.fp16:
        model.half()

    dataset = Dataset.from_json(args.test_file, args.batch_size, vocab=vocab,
                                random_unk=False,
                                oracle=args.strategy,
                                batch_group=args.batch_group,
                                batch_token_size=args.batch_token_size,
                                batch_action_size=args.batch_action_size,
                                max_length_diff=args.max_group_length_diff,
                                group_sentence_size=args.group_sentence_size)

    logger.info("model architecture")
    logger.info(model)
    model.eval()

    loss = eval_ppl(dataset, model, vocab)

    if args.save_to:
        with open(args.save_to, 'w', encoding='utf-8') as file:
            json.dump(loss, file)

    else:
        print(loss)


if __name__ == "__main__":
    arguments = parser.parse_args()
    main(arguments)
