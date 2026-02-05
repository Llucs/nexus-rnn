import re
import json
import os

class Tokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "[NAME]": 2}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "[NAME]"}
        self.vocab_size = 3

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"([?.!,])", r" \1 ", text)
        text = re.sub(r'[" "]+', " ", text)
        text = re.sub(r"[^a-zA-Z\u00C0-\u00FF?.!,]+", " ", text)
        return text.strip()

    def fit(self, texts):
        for text in texts:
            cleaned = self.clean_text(text)
            words = cleaned.split()
            for word in words:
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text, max_len=10):
        cleaned = self.clean_text(text)
        words = cleaned.split()
        tokens = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]
        
        # Padding or Truncating
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return tokens

    def decode(self, tokens):
        return " ".join([self.idx2word.get(t, "<UNK>") for t in tokens if t != 0])

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"word2idx": self.word2idx, "idx2word": {str(k): v for k, v in self.idx2word.items()}}, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.word2idx = data["word2idx"]
                self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
                self.vocab_size = len(self.word2idx)
