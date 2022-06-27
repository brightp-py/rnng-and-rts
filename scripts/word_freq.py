import re
import numpy as np
from collections import defaultdict

pattern = re.compile(r'(?<=\s)[^() ]+')
FILENAME = './rnng/data/train.txt'
SAVEFILE = './data/wordfreq.tsv'

with open(FILENAME, 'r', encoding='utf-8') as file:
    text = file.read()

freq = defaultdict(lambda: 0)

for word in pattern.findall(text):
    freq[word] += 1

single_occ = [word for word in freq if freq[word] == 1]
freq['<unk>'] = len(single_occ)

for word in single_occ:
    del freq[word]

vals = np.array(list(freq.values()))
vals = -np.log(vals / np.sum(vals))

with open(SAVEFILE, 'w', encoding='utf-8') as file:
    file.write('word\tfreq\n')
    file.write(
        '\n'.join(f"{word}\t{val}" for word, val in zip(freq.keys(), vals))
    )
