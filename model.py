import numpy as np
import pickle

class SimpleRNN:
    def __init__(self, vocab_size, hidden_size=256, output_size=None):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size else vocab_size

        # Weight Initialization
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01 
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01 
        self.Why = np.random.randn(self.output_size, hidden_size) * 0.01 
        
        self.bh = np.zeros((hidden_size, 1)) 
        self.by = np.zeros((self.output_size, 1)) 

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.last_inputs = inputs
        self.last_hs = { -1: h }
        
        for t, x_idx in enumerate(inputs):
            x = np.zeros((self.vocab_size, 1))
            x[x_idx] = 1
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            self.last_hs[t] = h
            
        y = np.dot(self.Why, h) + self.by
        exp_y = np.exp(y - np.max(y))
        probs = exp_y / np.sum(exp_y)
        return probs, self.last_hs

    def save_weights(self, path):
        weights = {
            'Wxh': self.Wxh, 'Whh': self.Whh, 'Why': self.Why,
            'bh': self.bh, 'by': self.by,
            'hidden_size': self.hidden_size
        }
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
            self.hidden_size = weights.get('hidden_size', 64)
            self.Wxh = weights['Wxh']
            self.Whh = weights['Whh']
            self.Why = weights['Why']
            self.bh = weights['bh']
            self.by = weights['by']
            self.vocab_size = self.Wxh.shape[1]
            self.output_size = self.Why.shape[0]
