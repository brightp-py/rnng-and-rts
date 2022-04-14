## Data file columns

* **item** (reading-time files) - The story that this data point was pulled from. **NOT** to be confused with...
* **item** (some RNNG surprisal files) - Simply the index of the datapoint in the file. This can almost always be ignored.
* **zone** (reading-time files) - The index of the word in whichever *item* (story) it appears in.
* **sent** (surprisal files) - The sentence this token appears in.
* **word_num** (surprisal files) - The index of this word in its sentence.
* **word** - Word as it appears in the text.
* **leaf_surp** (RNNG surprisal files) - RNNG surprisal of a word *ignoring* any syntactic instructions that precede it.
* **brn_surp** (RNNG surprisal files) - RNNG surprisal of a word *including* any parent syntactic nodes.
* **surp** (LSTM surprisal files) - LSTM surprisal of a word.

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