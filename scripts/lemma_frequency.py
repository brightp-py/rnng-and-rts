"""lemma_frequency.py.

Given a stimuli file, get the sentences and the frequency of each lemma.

Adapted from yasemingunal's code at:
https://github.com/UMWordLab/lexical-statistics/blob/main/lemma_freq_code.py

Brighton Pauli, 2022
"""

import xlrd
import pandas as pd
import csv

import nltk
from nltk.corpus import treebank
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer