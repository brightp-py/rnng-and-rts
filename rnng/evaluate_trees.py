#!/usr/bin/env python3
import sys
import os

import argparse

import torch
from torch import cuda
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
import logging

from train import create_model
from data import Dataset

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('--test_file', default='data/test.json')
parser.add_argument('--model_file', default='rnng.pt')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='If "cuda", GPU number --gpu is used.')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--strategy', default='top_down', choices=['top_down', 'in_order'])
parser.add_argument('--random_unk', action='store_true', help='Randomly replace a token to <unk> on training sentences by a probability inversely proportional to word frequency.')
parser.add_argument('--batch_token_size', type=int, default=15000, help='Number of tokens in a batch (batch_size*sentence_length) does not exceed this.')
parser.add_argument('--batch_action_size', type=int, default=45000, help='(batch_size*max_action_length) does not exceed this.')
parser.add_argument('--batch_group', choices=['same_length', 'random', 'similar_length', 'similar_action_length'],
                    default='similar_length', help='Sentences are grouped by this criterion to make each batch.')
parser.add_argument('--max_group_length_diff', default=20, type=int,
                    help='When --batch_group=similar_length or similar_action_length, maximum (token or action) length difference in a single batch does not exceed this.')
parser.add_argument('--group_sentence_size', default=1024, type=int,
                    help='When --batch_group=similar_length, sentences are first sorted by length and grouped by this number of sentences, from which each batch is sampled.')


# def do_valid(model, optimizer, scheduler, train_data, val_data, tb, epoch, step, val_losses, args):

#     val_loss, val_ppl, val_action_ppl, val_word_ppl = eval_action_ppl(val_data, model)
#     tb.write({'Valid ppl': val_ppl, 'Valid action ppl': val_action_ppl, 'Valid word ppl': val_word_ppl}, step)
#     tb.write({'Valid loss': val_loss}, use_time=True)

def load_model(checkpoint, action_dict, vocab):
  if 'model_state_dict' in checkpoint:
    model = create_model(checkpoint['args'], action_dict, vocab)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
  else:
    return checkpoint['model']

def eval_action_ppl(data, model, vocab):

    device = next(model.parameters()).device
    model.eval()

    losses = {}
    
    with torch.no_grad():
        for batch in data.batches():
            token_ids, action_ids, max_stack_size, subword_end_mask, batch_idx = batch

            # print(batch_idx)
            
            token_ids = token_ids.to(device)
            action_ids = action_ids.to(device)
            subword_end_mask = subword_end_mask.to(device)
            loss, a_loss, w_loss, _ = model(token_ids, action_ids,
                                            stack_size_bound=max_stack_size,
                                            subword_end_mask=subword_end_mask)
            
            start_a = 0
            start_w = 0

            for i, b in enumerate(batch_idx):

                b = int(b)
                losses[b] = {}

                last_action = action_ids[i].count_nonzero()
                losses[b]['actions'] = [
                    (model.action_dict.i2a[a], float(loss)) for a, loss in zip(
                        action_ids[i][:last_action],
                        a_loss[start_a:start_a+last_action]
                    )
                ]
                start_a += last_action

                nshifts = (action_ids[i]==1).count_nonzero()
                losses[b]['tokens'] = [
                    (vocab.id_to_word(w), float(loss)) for w, loss in zip(
                        token_ids[i][:nshifts],
                        w_loss[start_w:start_w+nshifts]
                    )
                ]
                start_w += nshifts

        print(losses)
            # break


def main(args):
    logger.info('Args: {}'.format(args))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    if args.device == 'cuda':
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    # load the model from its file
    checkpoint = torch.load(args.model_file)
    vocab = checkpoint['vocab']
    action_dict = checkpoint['action_dict']
    model = load_model(checkpoint, action_dict, vocab).to(device)

    # print(len(action_dict.i2a), action_dict.i2a)

    if args.fp16:
        model.half()

    # dataset = Dataset.from_json(args.test_file, args.batch_size, vocab, action_dict,
    #                                 prepro_args = prepro_args,
    #                                 batch_token_size = args.batch_token_size,
    #                                 batch_group = 'similar_length'
    # )

    dataset = Dataset.from_json(args.test_file, args.batch_size, vocab=vocab,
                                random_unk=args.random_unk,
                                oracle=args.strategy, batch_group=args.batch_group,
                                batch_token_size=args.batch_token_size,
                                batch_action_size=args.batch_action_size,
                                max_length_diff=args.max_group_length_diff,
                                group_sentence_size=args.group_sentence_size)

    # print(len(dataset))

    logger.info("model architecture")
    logger.info(model)
    model.eval()

    eval_action_ppl(dataset, model, vocab)

    # for batch in dataset.batches():
    #   token_ids, action_ids, max_stack_size, subword_end_mask, batch_idx = batch
    #   token_ids = token_ids.to(device)
    #   action_ids = action_ids.to(device)
    #   subword_end_mask = subword_end_mask.to(device)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)