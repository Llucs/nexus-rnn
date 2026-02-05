import numpy as np
import json
from tokenizer import Tokenizer
from model import SimpleRNN

def train():
    # 1. Load Data
    with open('data.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    pairs = [line.strip().split('|') for line in lines if '|' in line]
    inputs_raw = [p[0] for p in pairs]
    outputs_raw = [p[1] for p in pairs]

    # 2. Fit Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit(inputs_raw + outputs_raw)
    tokenizer.save('vocab.json')

    # 3. Prepare Training Data
    X = [tokenizer.encode(txt) for txt in inputs_raw]
    unique_responses = list(set(outputs_raw))
    resp2idx = {resp: i for i, resp in enumerate(unique_responses)}
    y = [resp2idx[resp] for resp in outputs_raw]
    
    with open('responses.json', 'w', encoding='utf-8') as f:
        json.dump(unique_responses, f)

    # 4. Initialize Model
    model = SimpleRNN(vocab_size=tokenizer.vocab_size, hidden_size=32, output_size=len(unique_responses))
    
    # 5. Training Loop
    learning_rate = 0.001
    epochs = 2000
    
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            # Forward
            probs, hs = model.forward(X[i])
            
            # Loss
            target_idx = y[i]
            loss = -np.log(probs[target_idx] + 1e-8)
            total_loss += loss
            
            # Backward
            dy = np.copy(probs)
            dy[target_idx] -= 1
            
            # Gradients
            dWhy = np.dot(dy, hs[len(X[i])-1].T)
            dby = dy
            dh = np.dot(model.Why.T, dy)
            dtanh = (1 - hs[len(X[i])-1] * hs[len(X[i])-1]) * dh
            dbh = dtanh
            dWxh = np.dot(dtanh, np.zeros((1, model.vocab_size))) # Placeholder for simplicity
            # Correct Wxh update
            x_idx = X[i][-1]
            x = np.zeros((model.vocab_size, 1))
            x[x_idx] = 1
            dWxh = np.dot(dtanh, x.T)

            # Update
            model.Why -= learning_rate * dWhy
            model.by -= learning_rate * dby
            model.bh -= learning_rate * dbh
            model.Wxh -= learning_rate * dWxh

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss[0]:.4f}")

    model.save_weights('weights.pkl')
    print("Training complete. Weights saved.")

if __name__ == "__main__":
    train()
