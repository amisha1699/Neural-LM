import re
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from task1_improved import WordPieceTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# WordPieceTokenizer class
# class WordPieceTokenizer:
#     def __init__(self):
#         self.vocab = {}

#     def preprocess_data(self, text):
#         """
#         Preprocess the text by converting to lowercase, removing special characters,
#         and splitting into words.
#         """
#         text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
#         text = re.sub(r"\s+", " ", text).strip()
#         return text.split()

#     def construct_vocabulary(self, corpus, vocab_file="vocabulary.txt"):
#         words = self.preprocess_data(corpus)

#         word_freq = {}
#         for word in words:
#             word_freq[word] = word_freq.get(word, 0) + 1

#         subwords = set(char for word in word_freq for char in word)
#         subwords.update(["##" + char for char in subwords])

#         for word, freq in word_freq.items():
#             if word not in subwords:
#                 for i in range(1, len(word)):
#                     subwords.add(word[:i])
#                     subwords.add("##" + word[i:])

#         self.vocab = {token: idx for idx, token in enumerate(sorted(subwords))}

#         with open(vocab_file, "w") as file:
#             for token in self.vocab:
#                 file.write(f"{token}\n")

#     def tokenize(self, sentence):
#         words = self.preprocess_data(sentence)
#         tokens = []

#         for word in words:
#             if word in self.vocab:
#                 tokens.append(word)
#             else:
#                 start = 0
#                 while start < len(word):
#                     for end in range(len(word), start, -1):
#                         subword = word[start:end]
#                         if (start > 0 and f"##{subword}" in self.vocab) or subword in self.vocab:
#                             tokens.append(f"##{subword}" if start > 0 else subword)
#                             start = end
#                             break
#                     else:
#                         tokens.append("[UNK]")
#                         break
#         return tokens

# Word2VecDataset class with negative sampling
class Word2VecDataset(Dataset):
    def __init__(self, corpus, tokenizer, context_window=2, negative_samples=5):
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.negative_samples = negative_samples
        self.data = self.preprocess_data(corpus)

    def preprocess_data(self, corpus):
        tokens = self.tokenizer.tokenize(corpus)
        token_indices = [self.tokenizer.vocab[token] for token in tokens if token in self.tokenizer.vocab]
        data = []

        for idx in range(self.context_window, len(token_indices) - self.context_window):
            context = token_indices[idx - self.context_window: idx] + token_indices[idx + 1: idx + self.context_window + 1]
            target = token_indices[idx]
            data.append((context, target))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# Word2Vec model with negative sampling
class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization

    def forward(self, context_words, target_word):
        # Compute the embeddings for the context words
        embedded = self.embeddings(context_words)
        context_vector = torch.mean(embedded, dim=1)  # Aggregate context word embeddings
        
        # Apply dropout
        context_vector = self.dropout(context_vector)
        
        # Compute the output
        output = self.linear(context_vector)
        
        # Compute the loss for the target word
        target_word_loss = nn.CrossEntropyLoss()(output, target_word)
        
        return target_word_loss

# from sklearn.metrics.pairwise import cosine_similarity

# Cosine similarity function to compute similarities between word embeddings
def compute_cosine_similarity(embeddings):
    """
    Compute cosine similarity between all pairs of embeddings.
    Args:
        embeddings (torch.Tensor): Tensor of shape (vocab_size, embedding_dim)
    Returns:
        similarity_matrix (torch.Tensor): Tensor of shape (vocab_size, vocab_size)
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    # Compute cosine similarity
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    return similarity_matrix

# Function to find triplets
def find_triplets(similarity_matrix, tokenizer, top_k=2):
    """
    Find triplets of tokens: two similar tokens and one dissimilar token.
    Args:
        similarity_matrix (torch.Tensor): Cosine similarity matrix of shape (vocab_size, vocab_size)
        tokenizer (WordPieceTokenizer): Tokenizer object to map indices to tokens
        top_k (int): Number of similar tokens to consider
    Returns:
        triplets (list): List of triplets, each containing two similar tokens and one dissimilar token
    """
    vocab_size = similarity_matrix.shape[0]
    triplets = []

    for i in range(vocab_size):
        # Get the most similar tokens (excluding itself)
        similar_indices = torch.topk(similarity_matrix[i], top_k + 1).indices[1:].tolist()
        # Get the least similar token
        dissimilar_index = torch.argmin(similarity_matrix[i]).item()

        # Map indices to tokens
        similar_tokens = [list(tokenizer.vocab.keys())[idx] for idx in similar_indices]
        dissimilar_token = list(tokenizer.vocab.keys())[dissimilar_index]

        # Add triplet to the list
        triplets.append((similar_tokens, dissimilar_token))

    return triplets



# Training function
def train(model, train_dataset, val_dataset, vocab_size, epochs=100, lr=0.001, patience=3):
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5)

    training_losses = []
    validation_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for context, target in train_dataloader:
            optimizer.zero_grad()
            context = context.cuda()
            target = target.cuda()

            # Pass both context and target to the model
            loss = model(context, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_dataloader)
        training_losses.append(average_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for context, target in val_dataloader:
                context = context.cuda()
                target = target.cuda()
                loss = model(context, target)
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(val_dataloader)
        validation_losses.append(average_val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}")

        # Early stopping and learning rate scheduling
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "word2vec_cbow_best.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        scheduler.step(average_val_loss)

    # Plotting the training and validation loss
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, len(training_losses) + 1), training_losses, marker='o', label='Training Loss')
    # plt.plot(range(1, len(validation_losses) + 1), validation_losses, marker='o', label='Validation Loss')
    # plt.title('Training and Validation Loss vs Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid()
    # plt.show()

# Main function
if __name__ == "__main__":
    # Example corpus
    try:
        with open("corpus.txt", "r") as file:
            corpus = file.read()
    except FileNotFoundError:
        print("Error: corpus.txt file not found.")
        exit()

    # Initialize tokenizer and construct vocabulary
    tokenizer = WordPieceTokenizer()
    tokenizer.construct_vocabulary(corpus)

    # Create dataset and split into training and validation sets
    dataset = Word2VecDataset(corpus, tokenizer, context_window=2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    vocab_size = len(tokenizer.vocab)
    embedding_dim = 10
    model = Word2VecModel(vocab_size, embedding_dim).cuda()  # Move model to GPU

    # Train the model
    train(model, train_dataset, val_dataset, vocab_size)
    embeddings = model.embeddings.weight.data.cpu()

    # Compute cosine similarity matrix
    similarity_matrix = compute_cosine_similarity(embeddings)

    # Find triplets
    triplets = find_triplets(similarity_matrix, tokenizer, top_k=2)

    # Print example triplets
    print("\nExample Triplets:")
    for i, triplet in enumerate(triplets[:2]):  # Print first two triplets
        similar_tokens, dissimilar_token = triplet
        print(f"Triplet {i + 1}:")
        print(f"  Similar Tokens: {similar_tokens}")
        print(f"  Dissimilar Token: {dissimilar_token}")
        print()