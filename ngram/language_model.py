#!/bin/env python
"""
language_model.py.

Written by joshualoehr.
https://github.com/joshualoehr/ngram-language-model
Edited by Brighton Pauli, 4/20/2022.
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import nltk

from preprocess import preprocess, EOS, UNK


def load_data(data_dir):
    """Load train and test corpora from a directory.

    Directory must contain two files: train.txt and test.txt.
    Newlines will be stripped out.

    Args:
        data_dir (Path) -- pathlib.Path of the directory to use.

    Returns:
        The train and test sets, as lists of sentences.

    """
    train_path = data_dir.joinpath('train.txt').absolute().as_posix()
    test_path = data_dir.joinpath('test.txt').absolute().as_posix()

    with open(train_path, 'r') as file:
        train_data = [line.strip() for line in file.readlines()]
    with open(test_path, 'r') as file:
        test_data = [line.strip() for line in file.readlines()]
    return train_data, test_data


class LanguageModel:
    """An n-gram language model trained on a given corpus.

    For a given n and given training corpus, constructs an n-gram language
    model for the corpus by:
    1. preprocessing the corpus (adding SOS/EOS/UNK tokens)
    2. calculating (smoothed) probabilities for each n-gram

    Also contains methods for calculating the perplexity of the model
    against another corpus, and for generating sentences.

    """

    def __init__(self, train_data, n_val, laplace=1):
        """Create a LanguageModel object.

        Args:
            train_data (list of str): list of sentences comprising the training
                corpus.
            n (int): the order of language model to build (i.e. 1 for unigram,
                2 for bigram, etc.).
            laplace (int): lambda multiplier to use for laplace smoothing
                (default 1 for add-1 smoothing).

        """
        self.n_val = n_val
        self.laplace = laplace
        self.tokens = preprocess(train_data, n_val)
        self.vocab = nltk.FreqDist(self.tokens)
        self.model = self._create_model()
        self.masks = list(reversed(list(product((0, 1), repeat=n_val))))

    def _smooth(self):
        """Apply Laplace smoothing to n-gram frequency distribution.

        Here, n_grams refers to the n-grams of the tokens in the training
        corpus, while m_grams refers to the first (n-1) tokens of each n-gram.

        Returns:
            dict: Mapping of each n-gram (tuple of str) to its Laplace-smoothed
                probability (float).

        """
        vocab_size = len(self.vocab)

        n_grams = nltk.ngrams(self.tokens, self.n_val)
        n_vocab = nltk.FreqDist(n_grams)

        m_grams = nltk.ngrams(self.tokens, self.n_val-1)
        m_vocab = nltk.FreqDist(m_grams)

        def smoothed_count(n_gram, n_count):
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            numer = (n_count + self.laplace)
            denom = (m_count + self.laplace * vocab_size)
            return numer / denom

        return {n_gram: smoothed_count(n_gram, count)
                for n_gram, count in n_vocab.items()}

    def _create_model(self):
        """Create a probability distribution for vocab of the training corpus.

        If building a unigram model, the probabilities are simple relative
        frequencies of each token with the entire corpus.

        Otherwise, the probabilities are Laplace-smoothed relative frequencies.

        Returns:
            A dict mapping each n-gram (tuple of str) to its probability
                (float).

        """
        if self.n_val == 1:
            num_tokens = len(self.tokens)
            return {(unigram,): count / num_tokens
                    for unigram, count in self.vocab.items()}
        return self._smooth()

    def _convert_oov(self, ngram):
        """Convert, if necessary, a given n-gram to one known by the model.

        Starting with the unmodified ngram, check each possible permutation of
        the n-gram with each index of the n-gram containing either the original
        token or <UNK>. Stop when the model contains an entry for that
        permutation.

        This is achieved by creating a 'bitmask' for the n-gram tuple, and
        swapping out each flagged token for <UNK>. Thus, in the worst case,
        this function checks 2^n possible n-grams before returning.

        Returns:
            The n-gram with <UNK> tokens in certain positions such that the
            model contains an entry for it.

        """
        def mask(ngram, bitmask):
            return tuple(
                token if flag else UNK for token, flag in zip(ngram, bitmask)
            )

        ngram = (ngram,) if isinstance(ngram, str) else ngram
        for possible_known in [mask(ngram, bitmask) for bitmask in self.masks]:
            if possible_known in self.model:
                return possible_known

        raise LookupError(f"Model failed to find n-gram {str(ngram)}.")

    def perplexity(self, test_data):
        """Calculate the perplexity of the model against a given test corpus.

        Args:
            test_data (list of str): sentences comprising the training corpus.
        Returns:
            The perplexity of the model as a float.

        """
        test_tokens = preprocess(test_data, self.n_val)
        test_ngrams = nltk.ngrams(test_tokens, self.n_val)
        total = len(test_tokens)

        known_ngrams = (self._convert_oov(ngram) for ngram in test_ngrams)
        probabilities = [self.model[ngram] for ngram in known_ngrams]

        return np.exp((-1/total) * sum(map(np.log, probabilities)))

    def sentence_surprisal(self, sent):
        """Return the surprisal for each token in the sentence.

        Args:
            sent (tuple OR str): sequence of words to get surprisals of.
        Returns:
            numpy array of the same length as sent, where each number
                corresponds to the surprisal of the token at the same index.
        """
        if isinstance(sent, str):
            sent = sent.split()
        probs = []
        prev = ["<s>"] * (self.n_val - 1)
        for word in sent:
            prev.append(word)
            key = self._convert_oov(prev)
            print('\t', key)
            probs.append(self.model[key])
            del prev[0]
        return -np.log(np.array(probs))

    def _best_candidate(self, prev, i, without=None):
        """Choose the most likely next token given the previous (n-1) tokens.

        If selecting the first word of the sentence (after the SOS tokens),
        the i'th best candidate will be selected, to create variety.
        If no candidates are found, the EOS token is returned with a
        probability of 1.

        Args:
            prev (tuple of str): the previous n-1 tokens of the sentence.
            i (int): which candidate to select if not the most probable one.
            without (list of str): tokens to exclude from the candidates list.
        Returns:
            A tuple with the next most probable token and its corresponding
                probability.

        """
        blacklist = ["UNK"]
        if without:
            blacklist += without
        candidates = ((ngram[-1], prob) for ngram, prob in self.model.items()
                      if ngram[:-1] == prev)
        candidates = filter(
            lambda candidate: candidate[0] not in blacklist, candidates)
        candidates = sorted(
            candidates, key=lambda candidate: candidate[1], reverse=True)
        if len(candidates) == 0:
            return (EOS, 1)
        return candidates[0 if prev != () and prev[-1] != "<s>" else i]

    def generate_sentences(self, num, min_len=12, max_len=24):
        """Generate num random sentences using the language model.

        Sentences always begin with the SOS token and end with the EOS token.
        While unigram model sentences will only exclude the UNK token, n>1
        models will also exclude all other words already in the sentence.

        Args:
            num (int): the number of sentences to generate.
            min_len (int): minimum allowed sentence length.
            max_len (int): maximum allowed sentence length.
        Yields:
            A tuple with the generated sentence and the combined probability
            (in log-space) of all of its n-grams.

        """
        for i in range(num):
            sent, total_prob = ["<s>"] * max(1, self.n_val-1), 1
            while sent[-1] != EOS:
                prev = () if self.n_val == 1 else tuple(sent[-(self.n_val-1):])
                blacklist = sent + ([EOS] if len(sent) < min_len else [])
                next_token, next_prob = self._best_candidate(
                    prev, i, without=blacklist)
                sent.append(next_token)
                total_prob *= next_prob

                if len(sent) >= max_len:
                    sent.append(EOS)

            yield ' '.join(sent), -1/np.log(total_prob)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("N-gram Language Model")
    parser.add_argument('--data', type=str, required=True,
                        help='Location of the data directory containing '
                             'train.txt and test.txt')
    parser.add_argument('--n', type=int, required=True,
                        help='Order of N-gram model to create (i.e. 1 for '
                             'unigram, 2 for bigram, etc.)')
    parser.add_argument('--laplace', type=float, default=0.01,
                        help='Lambda parameter for Laplace smoothing (default '
                             'is 0.01 -- use 1 for add-1 smoothing)')
    parser.add_argument('--num', type=int, default=10,
                        help='Number of sentences to generate (default 10)')
    args = parser.parse_args()

    # Load and prepare train/test data
    data_path = Path(args.data)
    train, test = load_data(data_path)

    print("Loading {}-gram model...".format(args.n))
    lm = LanguageModel(train, args.n, laplace=args.laplace)
    print("Vocabulary size: {}".format(len(lm.vocab)))

    # print("Generating sentences...")
    # for sentence, prob in lm.generate_sentences(args.num):
    #     print("{} ({:.5f})".format(sentence, prob))

    print("Generating surprisals...")
    sent1 = "I brought salt and pepper ."
    print(f"\t{sent1}")
    print('\t', lm.sentence_surprisal(sent1))
    sent2 = "I brought pepper and salt ."
    print(f"\t{sent2}")
    print('\t', lm.sentence_surprisal(sent2))

    perplexity = lm.perplexity(test)
    print("Model perplexity: {:.3f}".format(perplexity))
    print("")
