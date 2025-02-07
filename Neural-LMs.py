import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from WordPieceTokenizer import WordPieceTokenizer
from Word2Vec import Word2VecModel
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import os
import argparse

# # Function to check for pre-trained models in "saved_models/"
# def load_pretrained_model(model_name, model, device):
#     model_path = f"saved_models/{model_name}_final.pth"
#     if os.path.exists(model_path):
#         print(f"Loading pre-trained model: {model_name}")
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         model.to(device)
#     else:
#         print(f"Pre-trained model for {model_name} not found. Proceeding with training.")
#         return None 

# # Function to parse command-line arguments
# def pipeline_args():
#     """
#     Parses command-line arguments for the NLP pipeline.

#     Arguments:
#         -m or --model      : Specifies the model to train or use for prediction. Options include 'NeuralLM1', 'NeuralLM2', and 'NeuralLM3'.
#         -t or --test_file  : Specifies the path to the test file for prediction.

#     Returns:
#         argparse.Namespace: A namespace containing the parsed arguments.
#     """
#     parser = argparse.ArgumentParser(
#         description="Natural Language Processing Assignment 1\n"
#                     "Enter the following arguments:\n"
#                     "  1. -m or --model    : Choose a model for training or prediction (NeuralLM1, NeuralLM2, NeuralLM3).\n"
#                     "  2. -t or --test_file: Provide the path to the test file for prediction."
#     )
    
#     parser.add_argument('-m', '--model', type=str, required=True, help="Name of the model for training or prediction")
#     parser.add_argument('-t', '--test_file', type=str, required=True, help="Path to the test file for prediction")
    
#     return parser.parse_args()

class NeuralLMDataset(Dataset):
    def __init__(self, corpus, tokenizer, context_window):
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.data = self.preprocess_data(corpus)

    def preprocess_data(self, corpus):
        """
        Preprocess the corpus into context-target pairs for the next-three-token prediction task.
        """
        # Tokenize the corpus
        tokens = self.tokenizer.tokenize(corpus)
        token_indices = [self.tokenizer.vocab[token] for token in tokens if token in self.tokenizer.vocab]
        
        data = []
        for idx in range(self.context_window, len(token_indices) - 3):  
            context = token_indices[idx - self.context_window: idx]  
            target = token_indices[idx: idx + 3]  
            data.append((context, target))
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_tensor = torch.tensor(context, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return context_tensor, target_tensor

class NeuralLM1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NeuralLM1, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(embedding_dim * 5, hidden_dim)  
        self.output = nn.Linear(hidden_dim, vocab_size * 3)
    
    def forward(self, context):
        """
        Context shape: (batch_size, context_window)
        """
        embedded = self.embeddings(context)    
        context_vector = embedded.view(embedded.size(0), -1)          
        hidden_output = F.relu(self.hidden(context_vector))        
        output = self.output(hidden_output)  
        return output.view(output.size(0), 3, -1)

class NeuralLM2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NeuralLM2, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden1 = nn.Linear(embedding_dim * 5, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, vocab_size * 3)
    
    def forward(self, context):
        embedded = self.embeddings(context)
        context_vector = embedded.view(embedded.size(0), -1)
        hidden_output1 = F.leaky_relu(self.hidden1(context_vector))
        hidden_output2 = F.leaky_relu(self.hidden2(hidden_output1))
        output = self.output(hidden_output2)
        return output.view(output.size(0), 3, -1)

class NeuralLM3(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NeuralLM3, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim * 5, vocab_size * 3)
    
    def forward(self, context):
        embedded = self.embeddings(context).transpose(1, 2)  
        conv_out = F.relu(self.conv1(embedded))
        conv_out = F.relu(self.conv2(conv_out))
        flattened = conv_out.view(conv_out.size(0), -1)  
        output = self.fc(flattened)
        return output.view(output.size(0), 3, -1)
    
def train(model, train_dataloader, val_dataloader, epochs, batch_size, lr, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs("saved_models", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_acc = 0

        for context, target in train_dataloader:
            optimizer.zero_grad()
            context = context.to(device)
            target = target.to(device)
            output = model(context)  
            output = output.view(-1, output.size(-1))  
            target = target.view(-1)  
            loss = criterion(output, target)  
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += compute_accuracy(output, target)

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_acc = total_train_acc / len(train_dataloader)
        train_ppl = compute_perplexity(avg_train_loss)

        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        total_val_acc = 0

        with torch.no_grad():
            for context, target in val_dataloader:
                context, target = context.to(device), target.to(device)

                output = model(context)
                output = output.view(-1, output.size(-1))
                target = target.view(-1)

                loss = criterion(output, target)
                total_val_loss += loss.item()
                total_val_acc += compute_accuracy(output, target)

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_acc = total_val_acc / len(val_dataloader)
        val_ppl = compute_perplexity(avg_val_loss)

        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Train PPL: {train_ppl:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, Val PPL: {val_ppl:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increase patience counter
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
    torch.save(model.state_dict(), f"saved_models/{model_name}_epoch{epoch+1}.pth")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs Epochs")
    plt.legend()
    plt.grid()
    plt.show()

def compute_accuracy(predictions, targets):
    """
    Computes accuracy: (correctly predicted tokens / total tokens)
    """
    predicted_tokens = torch.argmax(predictions, dim=-1)  
    correct = (predicted_tokens == targets).sum().item()
    total = targets.numel()
    return correct / total

def compute_perplexity(loss):
    """
    Computes perplexity given the loss.
    """
    return math.exp(loss)

def predict(model, tokenizer, test_file, context_window=5, device=None):
    """
    Predicts the next words for given test sentences using a trained NeuralLM model.

    Parameters:
    - model: Trained NeuralLM model
    - tokenizer: Tokenizer used for tokenizing input text
    - test_file: Path to the test file containing input sentences
    - context_window: Number of context words used for prediction
    - device: Device to run the inference (CPU/GPU)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Read test sentences
    with open(test_file, 'r', encoding='utf-8') as file:
        test_sentences = file.readlines()

    predictions = []

    for test_sentence in test_sentences:
        test_sentence = test_sentence.strip()
        test_tokens = tokenizer.tokenize(test_sentence)
        test_token_indices = [tokenizer.vocab[token] for token in test_tokens if token in tokenizer.vocab]

        # Use the last `context_window` tokens as context
        context_indices = test_token_indices[-context_window:]
        if not context_indices:  # Skip empty sentences
            continue
        
        context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(context_tensor)

        # Get predicted word indices
        predicted_indices = output.argmax(dim=-1).squeeze().cpu().numpy()

        # Convert predicted indices back to words
        predicted_words = [list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(idx)]
                           for idx in predicted_indices]

        predictions.append((test_sentence, predicted_words))

    # Print predictions
    for sentence, predicted_words in predictions:
        print(f"Input: {sentence}")
        print(f"Predicted next words: {' '.join(predicted_words)}\n")

    return predictions  # Return predictions if needed for further processing


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args = pipeline_args()
    # model_name = args.model
    # test_file_path = args.test_file
    try:
        with open("corpus.txt", "r", encoding='utf-8') as file:
            corpus = file.read()
    except FileNotFoundError:
        print("Error: corpus.txt file not found.")
        exit()
        
    embedding_dim = 100
    hidden_dim = 256
    batch_size = 64
    learning_rate = 0.001
    epochs = 20
    context_window = 5
    patience = 5
    
    tokenizer = WordPieceTokenizer()
    try:
        with open("vocabulary1.txt", "r", encoding='utf-8') as file:
            vocab_lines = file.readlines()
            tokenizer.vocab = {line.strip(): idx for idx,
                               line in enumerate(vocab_lines)}
    except FileNotFoundError:
        print("Error: vocabulary1.txt file not found.")
        exit()
    word2vec_model = Word2VecModel(vocab_size=len(tokenizer.vocab), embedding_dim=embedding_dim)   
    word2vec_model.load_state_dict(torch.load("task2_Updated.pth", map_location=device))
    word2vec_model.to(device)
    
    dataset = NeuralLMDataset(corpus, tokenizer, context_window=5)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    models = {
        "NeuralLM1": NeuralLM1(vocab_size=len(tokenizer.vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim),
        "NeuralLM2": NeuralLM2(vocab_size=len(tokenizer.vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim),
        "NeuralLM3": NeuralLM3(vocab_size=len(tokenizer.vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    }
    
    # model = models.get(model_name)
    
    # if not model:
    #     print(f"Error: Model {model_name} not found!")
    #     exit()

    # # Check if the model has been pre-trained
    # model = load_pretrained_model(model_name, model, device)
    # if model is None:
    for model_name, model in models.items():
            print(f"\nTraining {model_name}...\n")

            # Move model to device
            model.to(device)

            # Initialize embeddings with pre-trained Word2Vec embeddings
            model.embeddings.weight.data.copy_(word2vec_model.embeddings.weight.data)
            model.embeddings.weight.requires_grad = True  # Allow fine-tuning

            # Train the model
            train(model, train_dataloader, val_dataloader, epochs=epochs, batch_size=batch_size, lr=learning_rate, patience=patience)
    test_file_path = "sample_test.txt"
    predictions = predict(model, tokenizer, test_file_path)