import os
from train import train
from chat import start_chat

def main():
    print("--- Nexus AI Project ---")
    
    if not os.path.exists('weights.pkl') or not os.path.exists('vocab.json'):
        print("Iniciando treinamento inicial...")
        train()
        print("-" * 20)
    
    start_chat()

if __name__ == "__main__":
    main()
