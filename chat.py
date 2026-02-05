import numpy as np
import json
import re
import time
from tokenizer import Tokenizer

class NexusChat:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.load('vocab.json')
        
        with open('data.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.pairs = [line.strip().split('|') for line in lines if '|' in line]
        self.user_name = None
        self.context_history = []

    def get_embedding(self, text):
        # Improved embedding: weighted frequency and keyword focus
        text = self.tokenizer.clean_text(text)
        tokens = self.tokenizer.encode(text)
        vec = np.zeros(self.tokenizer.vocab_size)
        
        # Keywords get more weight
        keywords = ["nome", "quem", "como", "onde", "porque", "qual", "nexus", "ajuda"]
        
        for t in tokens:
            if t != 0:
                weight = 2.0 if self.tokenizer.idx2word.get(t) in keywords else 1.0
                vec[t] += weight
                
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def extract_name(self, text):
        patterns = [
            r"meu nome é ([\w\s]+)",
            r"eu me chamo ([\w\s]+)",
            r"pode me chamar de ([\w\s]+)",
            r"sou o ([\w\s]+)",
            r"sou a ([\w\s]+)",
            r"me chamo ([\w\s]+)"
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                # Get the first word of the captured group as the name
                name = match.group(1).strip().split()[0].capitalize()
                return name
        return None

    def think(self, user_input):
        # Simulate a "thinking" process to understand intent
        print(f"\n[Nexus está pensando sobre: '{user_input}']")
        time.sleep(0.5) # Brief pause for realism
        
        # Intent detection logic
        if any(word in user_input.lower() for word in ["nome", "quem sou", "meu nome"]):
            print("[Intenção detectada: Identidade]")
        elif any(word in user_input.lower() for word in ["oi", "olá", "bom dia", "boa tarde", "boa noite"]):
            print("[Intenção detectada: Saudação]")
        elif any(word in user_input.lower() for word in ["quem é você", "seu nome", "o que você faz"]):
            print("[Intenção detectada: Auto-identificação]")
        else:
            print("[Intenção detectada: Conversa Geral]")

    def get_response(self, user_input):
        # 1. Thinking process
        self.think(user_input)
        
        # 2. Name extraction
        new_name = self.extract_name(user_input)
        if new_name:
            self.user_name = new_name
            
        # 3. Find best match using cosine similarity
        input_vec = self.get_embedding(user_input)
        best_sim = -1
        best_resp = "Desculpe, ainda estou aprendendo sobre isso. Pode me explicar melhor?"
        
        for q, a in self.pairs:
            q_vec = self.get_embedding(q)
            sim = np.dot(input_vec, q_vec)
            
            # Boost similarity if exact keywords match
            if any(word in user_input.lower() and word in q.lower() for word in ["nome", "nexus", "quem"]):
                sim += 0.2
                
            if sim > best_sim:
                best_sim = sim
                best_resp = a
        
        # 4. Personalization and Memory
        if "[NAME]" in best_resp:
            if self.user_name:
                best_resp = best_resp.replace("[NAME]", self.user_name)
            else:
                if any(phrase in best_resp for phrase in ["Seu nome é", "Você é"]):
                    best_resp = "Eu ainda não sei o seu nome. Como você se chama?"
                else:
                    best_resp = best_resp.replace("[NAME]", "amigo")
                    
        return best_resp

def start_chat():
    chat = NexusChat()
    print("\n" + "="*30)
    print("Nexus AI: Olá! Eu sou a Nexus. Como posso ajudar hoje?")
    print("="*30 + "\n")
    
    while True:
        try:
            user_input = input("Você: ")
        except EOFError:
            break
            
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("\nNexus AI: Foi um prazer conversar com você. Até logo!")
            break
            
        if not user_input.strip():
            continue
            
        response = chat.get_response(user_input)
        print(f"Nexus AI: {response}\n")

if __name__ == "__main__":
    start_chat()
