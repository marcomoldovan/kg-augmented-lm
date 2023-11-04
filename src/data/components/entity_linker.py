import os
import gzip
import json
import multiprocessing
import joblib
import numpy as np
import torch
import faiss
from marisa_trie import Trie
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# from src import utils


# log = utils.get_pylogger(__name__)

class EntityLinker:
    def __init__(
        self, 
        data_dir, 
        alias_file='wikidata5m_alias.tar.gz', 
        corpus_file='wikidata5m_text.txt.gz', 
        embeddings_file='wikidata5m_entity_embeddings.pkl', 
        use_embeddings=False, 
        pretrained_name='bert-base-uncased'):
        
        self.data_dir = data_dir
        
        alias_path = os.path.join(data_dir, alias_file)
        corpus_path = os.path.join(data_dir, corpus_file)
        embeddings_path = os.path.join(self.data_dir, embeddings_file)
        
        with gzip.open(corpus_path, 'rt', encoding='utf-8') as f:
            self.corpus = f.readlines()
            
        # Read the aliases file
        with gzip.open(alias_file, 'rt', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()[1:4813490]
        
        self.alias_to_id = {}
        for line in lines:
            aliases = line.strip().split('\t')
            for alias in aliases[1:]:
                self.alias_to_id[alias] = aliases[0]

        # Build the marisa-trie
        self.trie = Trie(self.alias_to_id.keys())
        
        self.use_embeddings = use_embeddings
        if use_embeddings:
            # Setup device and move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
            self.model = AutoModel.from_pretrained(pretrained_name).to(self.device)
            
            # Compute entity embeddings using a pretrained huggingface model
            self.entity_embeddings = self.compute_entity_embeddings(alias_path, embeddings_path)
                
            # Build Faiss index for efficient similarity search
            self.index = self.build_faiss_index(self.entity_embeddings)    
    
    
    def compute_entity_embeddings(self, alias_file, embeddings_file):
        # Check if embeddings are already present on disk
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                return joblib.load(f)

        entity_embeddings = {}
        with open(alias_file, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                entity_id = parts[0]
                all_embeddings = []
                for alias in parts[1:]:
                    sentence = alias  # Since aliases are often short, we treat them as sentences
                    inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    alias_embedding = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()
                    all_embeddings.append(alias_embedding)
                entity_embeddings[entity_id] = np.mean(all_embeddings, axis=0)
        
        # Save the embeddings to disk        
        with open(embeddings_file, 'wb') as f:
            joblib.dump(entity_embeddings, f)
                
        return entity_embeddings
    
    
    def build_faiss_index(self, entity_embeddings):
        embedding_matrix = np.array(list(entity_embeddings.values())).astype('float32')
        dimension = embedding_matrix.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_matrix)
        return index
    

    def compute_contextual_embedding(self, mention, full_text, start_pos, end_pos):
        # Extract the sentence containing the mention
        sentence_start = full_text.rfind('.', 0, start_pos) + 1  # Start after the last period before the mention
        sentence_end = full_text.find('.', end_pos)  # End at the next period after the mention
        sentence = full_text[sentence_start:sentence_end].strip()

        # Tokenize both the mention and the sentence
        mention_tokens = self.tokenizer.tokenize(mention)
        sentence_tokens = self.tokenizer.tokenize(sentence)

        # Find the start and end position of the mention in the sentence tokens
        mention_start_pos = None
        mention_end_pos = None
        for i in range(len(sentence_tokens) - len(mention_tokens) + 1):
            if sentence_tokens[i:i+len(mention_tokens)] == mention_tokens:
                mention_start_pos = i
                mention_end_pos = i + len(mention_tokens) - 1
                break

        # Convert tokens to inputs and pass through the model
        inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[0]

        # Extract the embeddings corresponding to the mention and average them
        mention_embedding = torch.mean(embeddings[mention_start_pos:mention_end_pos+1], dim=0)

        return mention_embedding.cpu().numpy()
    
    
    def find_best_entity_match(self, mention_embedding, k=1):
        distances, indices = self.index.search(mention_embedding.reshape(1, -1), k)
        closest_index = indices[0][0]
        entity_id = list(self.entity_embeddings.keys())[closest_index]
        return entity_id, distances[0][0]


    def link_entities(self, document):
        tokens = document.split()  # Basic whitespace-based tokenization
        i = 0
        entities_found = []

        while i < len(tokens):
            longest_match = None
            longest_match_length = 0
            current_string = tokens[i].lower()

            # Check for entity match starting from the current token
            if current_string in self.trie:
                entity_id = self.alias_to_id[current_string]
                longest_match = (entity_id, i, i)
                longest_match_length = 1

            # Extend the match to subsequent tokens to find longer matches
            j = i + 1
            while j < len(tokens) and current_string in self.trie:
                current_string += " " + tokens[j].lower()
                if current_string in self.trie:
                    entity_id = self.alias_to_id[current_string]
                    longest_match = (entity_id, i, j)
                    longest_match_length = j - i + 1
                j += 1

            # If a match was found, add it to entities_found and skip the matched tokens
            if longest_match:
                entities_found.append(longest_match)
                i += longest_match_length
            else:
                i += 1

        # Convert token positions to character positions
        char_entities_found = []
        char_pos = 0
        token_index = 0
        for entity in entities_found:
            while token_index < entity[1]:
                char_pos += len(tokens[token_index]) + 1  # +1 for space
                token_index += 1
            start_pos = char_pos
            while token_index <= entity[2]:
                char_pos += len(tokens[token_index]) + 1
                token_index += 1
            end_pos = char_pos - 2  # -1 for last space, -1 to get the end of the word
            char_entities_found.append((entity[0], start_pos, end_pos))

        if self.use_embeddings:
            for idx, (entity_id, start, end) in enumerate(char_entities_found):
                mention = document[start:end+1]
                mention_embedding = self.compute_contextual_embedding(mention)
                best_match, confidence = self.find_best_entity_match(mention_embedding)
                # Replace entity ID based on embedding match and add confidence
                char_entities_found[idx] = (best_match, start, end, confidence)

        return char_entities_found


    def link_entities_for_corpus(self, n_docs=None):
        if n_docs is not None:
            output_file = os.path.join(self.data_dir, f"wikidata5m_entities_{n_docs}.jsonl.gz")
        else:
            output_file = os.path.join(self.data_dir, "wikidata5m_entities.jsonl.gz")
        
        n_docs = n_docs or len(self.corpus)

        with gzip.open(output_file, 'wt') as f_out:
           for document in tqdm(self.corpus[:n_docs], desc="Linking entities for corpus"):
                    document_id, document_text = document.split('\t', 1)
                    entities = self.link_entities(document_text)
                    entry = {
                        "doc_id": document_id,
                        "entities": entities
                    }
                    f_out.write(json.dumps(entry) + "\n")

print(os.getcwd())
el = EntityLinker(data_dir="./data/wikidata5m")