# GPT-dev (Andrej Karpathy course)

Follow-along code and notes for two lectures in Andrej Karpathy's
"Neural Networks: Zero to Hero" series, 26 to 30 July 2024:

1. "Let's build GPT: from scratch, in code, spelled out."
   A character-level decoder Transformer on Tiny Shakespeare.
2. "Let's build the GPT Tokenizer."
   Byte-pair encoding (BPE), written by me first, then compared with the
   lecture's `minbpe`.

## What I did here

- Typed `bigram.py` and `gpt.py` along with the lecture and took notes on
  attention, the matrix-multiplication tricks, and layer normalisation.
- Trained the character-level GPT and generated 10,000 characters of
  Shakespeare-style text (bottom of the notebook).
- Implemented BPE training, encode, and decode myself before I watched the
  solution (`[personal implementation] (base tokenizer)`).
- Added the GPT-4 regex split pattern in a second version
  (`[personal implementation] V2 + regex`).
- Read the three papers in this folder:
  - "Attention Is All You Need" (Vaswani et al., 2017)
  - "Language Models are Unsupervised Multitask Learners" (GPT-2)
  - "Efficient Training of Language Models to Fill in the Middle"
    (arXiv 2207.14255)
- Saved the trained model to Hugging Face:
  [shng2025/gpt-dev__andrej-course](https://huggingface.co/shng2025/gpt-dev__andrej-course).

## Repository map

| File or folder | Content |
|---|---|
| `bigram.py` | Bigram baseline model from the lecture. |
| `gpt.py` | The decoder Transformer from the lecture. |
| `GPT-dev = Build from scratch.ipynb`, `(v2).ipynb` | Notebook versions with my notes and the 10k character sample. |
| `GPT-dev - Tokenizer (Follow Along).ipynb` | The tokenizer lecture, followed step by step. |
| `GPT-dev - Tokenizer [personal implementation] (base tokenizer).ipynb` | My own BPE tokenizer. |
| `GPT-dev - Tokenizer [personal implementation] V2 + regex.ipynb` | My BPE tokenizer with GPT-4 style regex pre-tokenisation. |
| `Original/`, `Original - Tokenizer/` | The lecture notebooks, unchanged. |
| `Notes links.md`, `Notes links - Tokenizer.md` | Links to the chat transcripts used while I studied. |

## Project timeline

1. [GPTesla](https://github.com/Ice-Citron/GPTesla): first model trained
   from scratch, a Python code generator.
2. **GPT-dev** (this repository): the fundamentals, from the lecture.
3. [nanoGPT-Valkyrie](https://github.com/Ice-Citron/nanoGPT-Valkyrie):
   GPT-2 124M reproduction and the first LayerNorm experiment.
4. [GPT-Valkyrie](https://github.com/Ice-Citron/GPT-Valkyrie): the research
   on LN, RMSN, and PN, with 24 published checkpoints and the paper.

## Licence

MIT. See [`LICENSE`](./LICENSE).
