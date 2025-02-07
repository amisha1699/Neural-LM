import re
import json
from collections import Counter


class WordPieceTokenizer:
    def __init__(self, vocab_size=None):
        self.vocab = {}
        self.vocab_size = vocab_size

    def preprocess_data(self, text):
        """
        Preprocess the text by converting to lowercase, removing special characters, 
        and splitting into words.
        """
        # Convert to lowercase and remove special characters except spaces
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    def construct_vocabulary(self, corpus, vocab_size=None, vocab_file="vocabulary1.txt"):
        """
        Construct vocabulary using a WordPiece-like approach and save to file.
        """
        # Preprocess the corpus
        words = self.preprocess_data(corpus)

        # Frequency count of words
        word_freq = Counter(words)

        # Initialize vocabulary with characters from all words
        subwords = set(char for word in word_freq for char in word)
        subwords.update(["##" + char for char in subwords])  # Subword markers

        # Iteratively build vocabulary
        for word, freq in word_freq.items():
            # If the word is new, add its subwords
            if word not in subwords:
                for i in range(1, len(word)):
                    subwords.add(word[:i])
                    subwords.add("##" + word[i:])

        # Sort vocabulary by frequency and size limit
        if vocab_size:
            sorted_subwords = sorted(subwords, key=lambda x: word_freq.get(
                x, 0), reverse=True)[:vocab_size]
        else:
            sorted_subwords = sorted(subwords)

        # Add special tokens
        self.vocab = {"[UNK]": 0, "[PAD]": 1}
        for idx, token in enumerate(sorted_subwords, start=2):
            self.vocab[token] = idx

        # Save vocabulary to file
        with open(vocab_file, "w") as file:
            for token in self.vocab:
                file.write(f"{token}\n")

    def tokenize(self, sentence):
        """
        Tokenize a given sentence using the constructed vocabulary.
        """
        # Preprocess input sentence
        words = self.preprocess_data(sentence)
        tokens = []

        for word in words:
            if word in self.vocab:
                tokens.append(word)
            else:
                # Break down unknown words into subwords
                start = 0
                temp_tokens = []
                while start < len(word):
                    for end in range(len(word), start, -1):
                        subword = word[start:end]
                        if (start > 0 and f"##{subword}" in self.vocab) or subword in self.vocab:
                            temp_tokens.append(
                                f"##{subword}" if start > 0 else subword)
                            start = end
                            break
                    else:
                        # Handle unknown characters as standalone tokens
                        temp_tokens.append("[UNK]")
                        break
                tokens.extend(temp_tokens)
        return tokens

    def tokenize_json(self, input_file="test.json", output_file="tokenized1.json"):
        """
        Tokenizes samples from a JSON file and saves the results to another JSON file.
        """
        try:
            with open(input_file, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            print(f"Error: {input_file} not found.")
            return

        tokenized_data = []

        # Iterate through the list of dictionaries in the input JSON
        for sample in data:
            sample_id = sample.get("id")
            sentence = sample.get("sentence")
            tokenized_data.append({
                "id": sample_id,
                "tokens": self.tokenize(sentence)
            })

        # Save tokenized data to JSON file
        with open(output_file, "w") as file:
            json.dump(tokenized_data, file, indent=4)
        print(f"Tokenized data saved to {output_file}")


if __name__ == "__main__":
    # Example corpus for vocabulary construction
    group_no = "01"
    try:
        with open("corpus.txt", "r") as file:
            corpus = file.read()
    except FileNotFoundError:
        print("Error: corpus.txt file not found.")
        exit()
    # Initialize the tokenizer with dynamic vocab size (e.g., 50)
    tokenizer = WordPieceTokenizer(vocab_size=25000)

    # Construct vocabulary with dynamic size
    tokenizer.construct_vocabulary(
        corpus, vocab_size=25000, vocab_file="vocabulary1.txt")
    print("Vocabulary saved to vocabulary1.txt")

    # Tokenize sentences from test.json
    tokenizer.tokenize_json(input_file="test.json",
                            output_file=f"tokenized_{group_no}.json")
