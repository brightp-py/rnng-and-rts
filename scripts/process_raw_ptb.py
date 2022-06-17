"""
process_raw_ptb.py.

This module is used to standardize raw Penn Treebank-style syntax trees. This
includes removing subtrees that include only undesired tokens.

ptb_file is the raw Penn Treebank-style data, as provided by the database.

token_file is the tsv-style .tok file of tokens, as provided by the database.

save_file is the file that the processed trees will be saved to. If no file is
    provided, defaults to the same as ptb_file, but with "raw" replaced with
    "processed" (or "-processed" appended to the end).

id_file is the file that the key between trees and tokens is saved to.

If start_fresh is selected, the program does NOT pick up where it left off and
    instead creates trees from the very beginning of the database. Honestly,
    this should be true by default.

Brighton Pauli
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
parser.add_argument('--id_file', default=None, help="*-ids.tsv")
parser.add_argument('--start_fresh', action='store_true')


def get_tokens(file_name: str):
    """Yield all tokens found at the given .tok file, and their items."""
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()[1:]
    for line in lines:
        yield remove_paren(line.split()[0]), int(line.split()[2])


class TokenHolder:
    """Iterate through trees and tokens and keeps track of indices."""

    HEADER = "index\tword\tsent\tsent_pos\tstory\tstory_pos\n"

    def __init__(self, token_file: str, data=None):
        """Create a token holder object with tokens from the given file.

        If some data has already been loaded, it should be included as a list
        of lists, where each internal list has the following elements:

        [
            Component text,
            Tree index,
            Word index in tree,
            Item in .tok,
            Zone in .tok
        ]
        """
        self._data = data if data else []
        self._token_iter = get_tokens(token_file)

        self.tree_index = 1
        self.tree_word = 0
        self.tok_zone = 0

        self.token, self.tok_item = next(self._token_iter)

    def save(self, save_file: str):
        """Save each component's indices to a tsv file."""
        with open(save_file, 'w', encoding='utf-8') as file:
            file.write(
                TokenHolder.HEADER +
                '\n'.join(
                    (str(i) + '\t' + '\t'.join(map(str, x)))
                    for i, x in enumerate(self._data)
                )
            )

    def next_token(self):
        """Save next token to the `token` attr and update token indices."""
        token, item = next(self._token_iter)
        if self.tok_item != item:
            self.tok_item = item
            self.tok_zone = 0
        self.tok_zone += 1
        self.token = token

    def save_component(self, cutoff=None):
        """Save the word at the current indices."""
        if cutoff is None:
            cutoff = len(self.token)

        self._data.append(
            [self.token[:cutoff], self.tree_index, self.tree_word,
             self.tok_item, self.tok_zone])
        self.token = self.token[cutoff:]

        if not self.token:
            self.next_token()

    def forest(self, trees: list[str]):
        """Iterate through all trees in the given list and update tree indices.

        This is needed because the raw ptb file actually splits by *sub*trees
        as well as complete trees.

        So this generator only yields a tree when it has a matching number of
        '(' and ')' tokens.
        """
        toret = ""
        for tree in trees:
            # [:-2] to remove ")" at end
            toret += tree[:-2]
            if toret.count('(') == toret.count(')'):
                yield toret
                self.tree_index += 1
                self.tree_word = 0
                toret = ""


def is_nonterminal(text: str):
    """Return True if string text starts with ( and doesn't end with )."""
    return text[0] == '(' and not text[-1] == ')'


def prune_tree(tree: str, holder: TokenHolder):
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

    stack = deque()

    for inst in instructions:
        # print(stack)
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
            if holder.token[:len(inst)] == inst:
                holder.save_component(len(inst))
                stack.append(inst)
                inst = ""
            else:
                while inst and holder.token == inst[:len(holder.token)]:
                    inst = inst[len(holder.token):]
                    stack.append(holder.token)
                    holder.save_component()
            if not inst:
                holder.tree_word += 1

    if not stack:
        return ""

    return stack.pop()


def remove_top(tree: str):
    """Remove the top node of the tree."""
    start = tree.find('(', 1)
    return tree[start:-1]


def remove_paren(token: str):
    """Remove ( and ) from the given token."""
    return token.replace('(', '') \
                .replace(')', '')


def fix_quotes(tree: str):
    """Replace all quotes in the tree with single apostrophes."""
    return tree.replace("''", "'") \
               .replace("``", "'") \
               .replace('"', "'")


def save(trees, tree_file):
    """Save trees (which must be strings) to the given tree file."""
    print(len(trees))
    with open(tree_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(trees))


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

    holder = TokenHolder(args.token_file)

    for tree in holder.forest(trees):
        tree = fix_quotes(tree)
        processed = prune_tree(tree, holder)

        processed = f"(* {processed})"
        assert processed.count('(') == processed.count(')')
        done.append(processed)

        if not len(done) % 16:
            save(done, save_to)
            holder.save(args.id_file)

    save(done, save_to)
    holder.save(args.id_file)


if __name__ == "__main__":
    arguments = parser.parse_args()
    main(arguments)
