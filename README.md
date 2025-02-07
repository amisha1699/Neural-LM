# Neural-LM

## Overview
This project consists of three tasks that involve implementing fundamental NLP techniques from scratch. The tasks include building a WordPiece Tokenizer, training a Word2Vec model using the CBOW approach, and developing a Neural Language Model (MLP-based) with multiple variations. The implementation strictly avoids using external NLP libraries and instead relies on Python, NumPy, pandas, and PyTorch.

## Table of Contents
- Task 1: Implement WordPiece Tokenizer
- Task 2: Implement Word2Vec
- Task 3: Train a Neural LM
- Results
- Conclusion

## Task 1: WordPiece Tokenizer
### Objective
To implement a WordPiece Tokenizer from scratch without using any external NLP libraries.

### Implementation Details
`WordPieceTokenizer` class -
-    `preprocess_data()`
	    -   Handles all necessary preprocessing of the input text data.
	    -   Standard text cleaning techniques are applied without using lemmatization or stemming.
- `construct_vocabulary()`
	- Generates a vocabulary from the provided corpus.
- `tokenize`
	- Tokenizes a given sentence into a list of subword tokens based on the constructed vocabulary.

### Output
-   A text file containing the learned vocabulary.
-   A tokenizer capable of converting raw text into WordPiece tokens.

## Task 2: Word2Vec
### Objective

To build a Word2Vec model using the CBOW (Continuous Bag of Words) approach entirely from scratch in PyTorch.

### Implementation Details

`Word2VecDataset` class

-   Prepares CBOW training data from the text corpus.
    
-   Uses the `WordPieceTokenizer` from Task 1 for tokenization.
    
-   Compatible with PyTorch’s `DataLoader` for efficient batch processing.
    

### Word2VecModel Class

-   Implements the CBOW architecture using an embedding layer and a simple feed-forward network.
    
-   Trained using PyTorch’s autograd functionality.
    

### `train()` Function

-   Manages the entire training process, including forward pass, loss computation, backpropagation, and model checkpoint saving.
    

## Output

-   A trained Word2Vec model checkpoint (`task2_modelCheck.pth`) for use in later tasks.
    
-   Word embeddings learned from the corpus.


## Task 3: Train a Neural LM

### Objective

To train a Neural Language Model (MLP-based) using PyTorch, leveraging the previously implemented tokenizer and word embeddings.

### Implementation Details

`NeuralLMDataset` class

-   Prepares data for next-word prediction.
    
-   Uses `WordPieceTokenizer` and `Word2VecModel` for tokenization and embedding generation.
    
-   Compatible with PyTorch’s `DataLoader`.
    

### Three Variations of Neural LM

**NeuralLM1**

- Simple MLP with a single hidden layer using ReLU activation. - Embedding layer followed by a fully connected layer. - Output is reshaped to predict three tokens per input context.

**NeuralLM2**

- Deeper MLP with two hidden layers for improved feature extraction. - Uses Leaky ReLU activation for better gradient flow. - Second hidden layer reduces dimensionality before output.

**NeuralLM3**

- CNN-based model using 1D convolutions to capture local dependencies. - Two convolutional layers extract hierarchical features. - Fully connected layer maps extracted features to output tokens.
    

### `train()` Function

-   Handles the training of all three Neural LM architectures.
    
-   Computes accuracy and perplexity for model evaluation.
    

### Output

-   Three trained Neural Language Models with different architectures.
    
-   Accuracy and perplexity scores for performance comparison.

## Results

| ModelArchitecture | Details | Accuracy | Perplexity | Rationale |
|---|---|---|---|---|
| NeuralLM1 | Basic neural network with one hidden layer, ReLU activation, and fully connected layers. | 0.70 | 3.16 | Simple architecture, fails to capture complex dependencies. |
| NeuralLM2 | Deeper architecture with two hidden layers and Leaky ReLU activation. | 0.56 | 5.81 | Added complexity may have led to overfitting, resulting in performance drop. |
| NeuralLM3 | Convolutional layers with ReLU activation, followed by a fully connected layer for output. | 0.90 | 1.40 | Conv layers generalize well for next token prediction task as they can learn local dependencies. |

## Conclusion
This project successfully implements fundamental NLP techniques from scratch. The WordPiece Tokenizer efficiently tokenizes text, the Word2Vec model learns meaningful word representations, and the Neural Language Model predicts next-word tokens with varying degrees of complexity.