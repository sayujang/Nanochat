# NanoGPT: Character-Level Transformer Language Model

A PyTorch implementation of a Decoder-only Transformer model designed for character-level text generation. This model is capable of learning the statistical patterns of a text dataset (like Shakespeare) and generating new, coherent text that mimics the style of the input.

## Overview

This project implements a **Generative Pre-trained Transformer (GPT)** from scratch. Unlike simple Bigram models, this architecture uses **Self-Attention** to allow tokens to "talk" to each other, capturing long-range dependencies in the text.

### Key Features

* **Multi-Head Self-Attention:** Allows the model to focus on different parts of the input sequence simultaneously.
* **Feed-Forward Networks:** Processes information individually at each token position.
* **Residual Connections & Layer Normalization:** Ensures stable training and efficient gradient flow deep into the network.
* **Positional Embeddings:** Learnable embeddings that let the model understand the order of characters.
* **Dropout:** Regularization to prevent overfitting.

## Architecture

The model follows the standard GPT architecture:

1. **Input:** Character indices.
2. **Embedding Layer:** Token Embeddings + Positional Embeddings.
3. **Transformer Blocks:** A stack of `N` blocks, each containing:
* Masked Multi-Head Attention (Communication phase).
* Feed-Forward Network (Computation phase).
* LayerNorm (Pre-norm formulation).


4. **Output Head:** Linear projection to vocabulary size.

## Getting Started

### Prerequisites

* Python 3.x
* PyTorch (`pip install torch`)

### Installation

1. Clone this repository.
2. Ensure you have a text file named `input.txt` in the root directory. This will be your training data (e.g., the complete works of Shakespeare).

### Usage

Run the script to start training:

```bash
python train.py

```

*Note: Replace `train.py` with whatever you named your python file.*

### What happens when you run it?

1. The script reads `input.txt`.
2. It builds a vocabulary of unique characters.
3. It trains the Transformer model for `5000` iterations.
4. Every `500` iterations, it prints the current Training and Validation loss.
5. Finally, it generates **1000 characters** of text based on what it learned and decodes it to the console.

## Configuration & Hyperparameters

You can tweak the hyperparameters at the top of the script to fit your GPU memory or dataset size:

| Parameter | Value | Description |
| --- | --- | --- |
| `batch_size` | 64 | How many independent sequences are processed in parallel. |
| `block_size` | 256 | The maximum context length (time steps) for prediction. |
| `num_iter` | 5000 | Total number of training steps. |
| `learning_rate` | 3e-4 | The step size for the AdamW optimizer. |
| `n_embed` | 384 | The dimension of the embedding vectors. |
| `n_head` | 6 | Number of attention heads (384/6 = 64 dim per head). |
| `n_layer` | 6 | *Implicitly defined by the Sequential block in BigramModel.* |
| `dropout` | 0.2 | Probability of dropping neurons during training. |

## Code Structure

* **`Head`**: A single self-attention head.
* **`MultiHead`**: Runs multiple `Head` instances in parallel and concatenates the results.
* **`FeedForward`**: A simple multi-layer perceptron (MLP) with ReLU activation.
* **`Block`**: Combines Attention and FeedForward with Residual connections and LayerNorm.
* **`BigramModel`**: The main class (despite the name, it is a full Transformer) that assembles the blocks and handles embeddings.

## Sample Output

*After training on Shakespeare for 5000 iterations, the model might output something like:*

> DUKE VINCENTIO:
> I will be the one that shall be saved.
> ISABELLA:
> The heavens grant that implies the king!

*(Note: Actual output depends heavily on your `input.txt` dataset and training duration.)*

## Acknowledgements

This code is based on the concepts of the "Attention Is All You Need" paper and inspired by Andrej Karpathy's "NanoGPT" lecture series.
