"""
process_raw_ptb.py.

This module is used to standardize raw Penn Treebank-style syntax trees. This
includes removing subtrees that include only undesired tokens.
"""

import os

import argparse
from collections import deque

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--ptb_file', default='D:/data/naturalstories/ptb/raw.txt')
parser.add_argument('--token_file',
                    default='D:/data/naturalstories/all_stories.tok')
parser.add_argument('--save_file', default='')
parser.add_argument('--id_file', default=None, help="*-ids.csv")
parser.add_argument('--start_fresh', action='store_true')


def is_nonterminal(text: str):
    """Return True if string text starts with ( and doesn't end with )."""
    return text[0] == '(' and not text[-1] == ')'


def prune_tree(tree: str, tokens_iter, token):
    """Remove any terminals not in tokens and prune leafless branches.

    Parameters:
        tree   - Penn Treebank-style tree.

        tokens - List of tokens that should appear in the resulting tree.
               | * Excluding the first token.

        token  - The first token that should appear in the resulting tree.

    Returns a Penn Treebank-style tree with the same nonterminals as a string,
    the next token to appear in the iterator, the entire sentence with each
    token separated by a space, and the item of text that the final token
    appears in (as a string id).

    (S (NP *) (V Believe)) ==> (S (V Believe))

    >>> prune_tree("(S (NP Me))", iter(["Me"]))
    '(S (NP Me))'
    >>> prune_tree("(S (NP *) (V Believe))", iter(["Believe"]))
    '(S (V Believe))'
    >>> prune_tree("(S (NP The    boy))", iter(["The", "boy"]))
    '(S (NP The boy))'
    >>> text = "(SBAR (WHNP-1 (WP who)) (S (NP (-NONE- *T*-1)) (VP (VB saw))))"
    >>> prune_tree(text, iter(["who", "saw"]))
    '(SBAR (WHNP-1 (WP who)) (S (VP (VB saw))))'
    >>> prune_tree("(S (NP I,))", iter(["I", ","]))
    '(S (NP I ,))'
    >>> prune_tree("(S (NP I ,))", iter(["I,"]))
    '(S (NP I ,))'
    """
    instructions = tree.replace(')', ' ) ').split()

    sentence = []
    stack = deque()

    for inst in instructions:
        if inst == ')':
            comp = ""
            prev = stack.pop()
            while not is_nonterminal(prev):
                comp = ' ' + prev + comp
                prev = stack.pop()
            if comp:
                stack.append(prev + comp + ')')
        elif is_nonterminal(inst):
            stack.append(inst.split('-')[0])
        else:
            if token[:len(inst)] == inst:
                token = token[len(inst):]
                if not token:
                    try:
                        token, text_item = next(tokens_iter)
                    except StopIteration:
                        token = ""
                sentence.append(inst)
                stack.append(inst)
            else:
                while token and token == inst[:len(token)]:
                    inst = inst[len(token):]
                    sentence.append(token)
                    stack.append(token)
                    try:
                        token, text_item = next(tokens_iter)
                    except StopIteration:
                        token = ""
                        break

    if not stack:
        return "", token

    return stack.pop(), token, ' '.join(sentence), text_item


def forest(trees):
    """Iterate through all trees in the given list.
    
    This is needed because the raw ptb file actually splits by *sub*trees as
    well as complete trees.

    So this generator only yields a tree when it has a matching number of '('
    and ')' tokens.
    """
    toret = ""
    for tree in trees:
        # [:-2] to remove ")" at end
        toret += tree[:-2]
        if toret.count('(') == toret.count(')'):
            yield toret
            toret = ""


def remove_top(tree: str):
    """Remove the top node of the tree."""
    start = tree.find('(', 1)
    return tree[start:-1]


def remove_paren(token: str):
    """Remove ( and ) from the given token."""
    return token.replace('(', '') \
                .replace(')', '')


def get_tokens(file_name):
    """Yield all tokens found at the given .tok file."""
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()[1:]
    for line in lines:
        yield remove_paren(line.split()[0]), line.split()[2]


def fix_quotes(tree: str):
    """Replace all quotes in the tree with single apostrophes."""
    return tree.replace("''", "'") \
               .replace("``", "'") \
               .replace('"', "'")


def save(trees, ids, tree_file, id_file):
    print(len(trees))
    with open(tree_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(trees))
    
    if id_file:
        with open(id_file, 'w', encoding='utf-8') as file:
            file.write('\n'.join(','.join(row) for row in ids))


def main(args):
    """Standardize raw PTB-style trees and save the results."""
    with open(args.ptb_file, 'r', encoding='utf-8') as file:
        trees = file.read().split('(ROOT')[1:]  # [1:] to remove "" at start

    if args.save_file:
        save_to = args.save_file
    elif 'raw' in args.ptb_file:
        file_name = os.path.basename(args.ptb_file)
        file_name = file_name.replace('raw', 'processed')
        save_to = os.path.join(os.path.dirname(args.ptb_file), file_name)
    else:
        file_name = os.path.basename(args.ptb_file)
        file_name = f"{file_name.split('.')[0]}-processed.txt"
        save_to = os.path.join(os.path.dirname(args.ptb_file), file_name)

    done = []
    if os.path.exists(save_to) and not args.start_fresh:
        with open(save_to, 'r', encoding='utf-8') as file:
            done = file.read().split('\n')
        if len(done) < len(trees):
            trees = trees[len(done):]
        else:   # restart
            done = []

    tokens = get_tokens(args.token_file)
    token, text_item = next(tokens)

    sentences = []
    sentence_id = 0

    for tree in forest(trees):
        tree = fix_quotes(tree)
        processed, token, sentence, text_item = prune_tree(tree, tokens, token)

        sentence_id += 1
        
        sentences.append((text_item, str(sentence_id), sentence))

        processed = f"(* {processed})"
        assert(processed.count('(') == processed.count(')'))
        done.append(processed)

        if not len(done) % 16:
            save(done, sentences, save_to, args.id_file)

    save(done, sentences, save_to, args.id_file)


if __name__ == "__main__":
    arguments = parser.parse_args()
    main(arguments)
