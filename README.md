## Natural Stories Corpus

This research project makes use of the Natural Stories Corpus, a database of 
reading times gathered from human participants reading stories. The corpus 
makes use of rare grammatical constructions to ensure variety and test models' 
syntactic capabilities.

This corpus can be found at https://github.com/languageMIT/naturalstories.

Richard Futrell, et al. "The Natural Stories corpus: A reading-time corpus of English texts containing rare syntactic constructions". Language Resources and Evaluation 55. 1(2021): 63–77.

## Data file columns

* **story** - The story that this data point was pulled from.
* **story** - The index of the word in whichever *story* it appears in.
* **sent** - The sentence this token appears in.
* **sent_pos** - The index of this word in its sentence.
* **word** - Word as it appears in the text.
* **leaf_surp** - RNNG surprisal of a word *ignoring* any syntactic instructions that precede it.
* **branch_surp** - RNNG surprisal of a word *including* any parent syntactic nodes.
* **lstm_surp** - LSTM surprisal of a word.

## Files added or edited by me

<pre>
rnng
├─ <b>evaluate_trees.py</b>
└─ preprocess.py

lstm
├─ src
|  └─ language_models
|     └─ <b>evaluate_all_words.py</b>
└─ <b>lm</b>

<b>analysis</b>

<b>data</b>

<b>scripts</b>
├─ <b>component_mean.py</b>
├─ <b>conjoin_tables.py</b>
├─ <b>process_raw_ptb.py</b>
├─ <b>tree_to_seq.py</b>
└─ <b>vis_scatter.py</b>
</pre>

**Bolded** files and directories were created.