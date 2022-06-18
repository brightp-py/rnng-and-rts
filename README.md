## Data file columns

* **story** - The story that this data point was pulled from.
* **story** - The index of the word in whichever *story* it appears in.
* **sent** - The sentence this token appears in.
* **sent_pos** - The index of this word in its sentence.
* **word** - Word as it appears in the text.
* **leaf_surp** - RNNG surprisal of a word *ignoring* any syntactic instructions that precede it.
* **branch_surp** - RNNG surprisal of a word *including* any parent syntactic nodes.
* **lstm_surp** - LSTM surprisal of a word.

## LSTM

The implementation for this project's LSTM (Long-Short Term Memory) model can 
be found at https://github.com/facebookresearch/colorlessgreenRNNs.

K. Gulordava, P. Bojanowski, E. Grave, T. Linzen, M. Baroni. 2018. [Colorless green recurrent networks dream hierarchically.](https://arxiv.org/abs/1803.11138) Proceedings of NAACL. [[bib]](https://aclanthology.coli.uni-saarland.de/papers/N18-1108/n18-1108.bib) [[aclweb]](https://aclanthology.coli.uni-saarland.de/papers/N18-1108/n18-1108)

## RNNG

The implentation for this project's RNNG (Recurrent Neural Network Grammar) 
can be found at https://github.com/aistairc/rnng-pytorch.

Noji, Hiroshi, and Yohei Oseki. "Effective batching for recurrent neural network grammars." arXiv preprint arXiv:2105.14822 (2021).

## URNNG

The implentation for this project's URNNG (Unsupervised Recurrent Neural 
Network Grammar) can be found at https://github.com/harvardnlp/urnng.

Kim, Yoon, et al. "Unsupervised recurrent neural network grammars." arXiv preprint arXiv:1904.03746 (2019).

## Natural Stories Corpus

This research project makes use of the Natural Stories Corpus, a database of 
reading times gathered from human participants reading stories. The corpus 
makes use of rare grammatical constructions to ensure variety and test models' 
syntactic capabilities.

This corpus can be found at https://github.com/languageMIT/naturalstories.

Richard Futrell, et al. "The Natural Stories corpus: A reading-time corpus of English texts containing rare syntactic constructions". Language Resources and Evaluation 55. 1(2021): 63–77.

## Files added or edited by me

<pre>
lstm
├─ src
|  └─ language_models
|     └─ <b>evaluate_all_words.py</b>
└─ <b>lm</b>

rnng
├─ scripts
|  └─ <b>pull_from_json.py</b>
├─ <b>evaluate_trees.py</b>
└─ preprocess.py

<b>analysis</b>

<b>data</b>

<b>imgs</b>

<b>scripts</b>
</pre>

**Bolded** files and directories were created.

## License

The LSTM code is licensed under CC-BY-NC license. See the 
[lstm/LICENSE](https://github.com/brightp-py/rnng-and-rts/blob/main/lstm/LICENSE) 
file for more details.

Else, MIT