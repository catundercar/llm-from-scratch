# Phase 1: Text and Data -- From Characters to Training Batches

## Overview

Before you can train a language model, you need to solve two fundamental problems:
how do you represent text as numbers, and how do you feed those numbers to a model
efficiently? This phase tackles both.

## 1. Tokenization: Why and How

### The Problem

Neural networks operate on numbers, not characters. The simplest approach -- mapping
each character to an integer -- works but is inefficient. English text uses roughly
100 distinct characters, but a character-level model must process very long sequences
to capture meaning. The word "understanding" is 13 characters, forcing the model to
learn relationships across 13 time steps for a single concept.

Word-level tokenization goes to the other extreme: each word is one token, but the
vocabulary explodes (hundreds of thousands of entries), and the model cannot handle
words it has never seen.

### Byte Pair Encoding (BPE)

BPE finds a middle ground. The algorithm is elegant:

1. Start with a vocabulary of individual characters.
2. Count every adjacent pair of tokens in the corpus.
3. Merge the most frequent pair into a single new token.
4. Repeat steps 2-3 until you reach the desired vocabulary size.

After training, common words like "the" become single tokens, while rare words are
broken into familiar subword pieces. The word "unhappiness" might become
["un", "happiness"] or ["un", "happi", "ness"], depending on the training corpus.

### Encoding and Decoding

Once merges are learned, encoding new text applies the same merges in the same
priority order. Decoding simply concatenates the string representations of each
token ID.

Key insight: the order of merges matters. Merge 1 was learned first because it was
the most frequent pair in the original character-level representation. When encoding,
you must apply merges in the same order they were learned.

## 2. Data Loading: Sliding Windows

### Next-Token Prediction

Language models learn by predicting the next token. Given the sequence
[A, B, C, D, E], the model should predict:
- Given [A] -> predict B
- Given [A, B] -> predict C
- Given [A, B, C] -> predict D
- Given [A, B, C, D] -> predict E

In practice, we limit the context to a fixed window size called `block_size`.
With block_size=4, we create training examples:

| Input (x)       | Target (y)      |
|------------------|-----------------|
| [A, B, C, D]    | [B, C, D, E]    |
| [B, C, D, E]    | [C, D, E, F]    |

Notice that `y` is simply `x` shifted right by one position.

### Batching

PyTorch's DataLoader handles shuffling and batching. We wrap our sliding-window
data in a Dataset class, and the DataLoader produces batches of shape
`(batch_size, block_size)`.

### Train/Validation Split

We split the tokenized text into training and validation sets (typically 90/10).
The model trains on the training set, and we monitor loss on the validation set
to detect overfitting.

## 3. What You Will Build

In this phase you will implement:

1. **A BPE Tokenizer** -- Train it on text, encode strings to integer sequences,
   decode them back. You will also implement save/load for persistence.

2. **A Sliding-Window DataLoader** -- Create PyTorch Dataset and DataLoader objects
   that produce (input, target) pairs from tokenized text.

## Key Concepts to Remember

- **Subword tokenization** balances vocabulary size against sequence length.
- **BPE merge order** is the key to consistent encoding.
- **Next-token prediction** is the core training objective for autoregressive LMs.
- **Sliding window** creates overlapping training examples from a single sequence.
- **Train/val split** prevents overfitting and gives you an honest loss metric.
