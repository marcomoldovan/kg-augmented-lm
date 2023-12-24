import os
import gzip
import json
import joblib
import pickle
import numpy as np
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# from src.data.components.entity_linker import EntityLinker

def prune_corpus(data_dir: str, tokenizer: AutoTokenizer, min_doc_len: int = 256):
    corpus_path = os.path.join(data_dir, 'wikidata5m_text.txt.gz')
    corpus_path_pruned = corpus_path.replace('.txt.gz', f'_min-{min_doc_len}.txt.gz')
    
    if not os.path.exists(corpus_path_pruned):
        with gzip.open(corpus_path, 'rt') as f_in, gzip.open(corpus_path_pruned, 'wt') as f_out:
            for line in tqdm(f_in, desc="Pruning Corpus", unit=" lines"):
                # Splitting the line into id and text
                id, text = line.strip().split('\t', 1)
                
                # Tokenize the text and check its length
                token_ids = tokenizer(text)['input_ids']
                if len(token_ids) >= min_doc_len:
                    # Write the valid line to the new file
                    f_out.write(f"{id}\t{text}\n")
            

def whitelist_nq_entities(entity_linker, data_dir: str):
    nq_path = os.path.join(data_dir, '..', 'natural_questions/simplified-nq-train.jsonl.gz')
    whitelist_path = os.path.join(data_dir, 'wikidata5m_whitelist_entities.txt.gz')
    
    if not os.path.exists(whitelist_path):
        with gzip.open(nq_path, 'rt', encoding='utf-8') as f:
            single_entity_answers = 0
            multiple_entity_answers = 0
            one_node_short_answers = []
            multiple_node_short_answers = []
            one_node_short_answer_entities = []
            multiple_node_short_answer_entities = []
            for i, line in enumerate(f):
                entry = json.loads(line)
                
                short_answer = entry['annotations'][0]['short_answers']
                long_answer = entry['annotations'][0]['long_answer']['candidate_index']
                
                if (short_answer != []) and (long_answer != -1):
                    answer = " ".join(entry['document_text'].split(' ')[short_answer[0]['start_token']:short_answer[0]['end_token']])
                    entities = entity_linker.link_entities(answer)
                    if len(entities) == 1:
                        single_entity_answers += 1
                        one_node_short_answers.append(answer)
                        one_node_short_answer_entities.append(entities[0][0])
                    elif len(entities) > 1:
                        multiple_entity_answers += 1
                        multiple_node_short_answers.append(answer)
                        multiple_node_short_answer_entities.extend([entity[0] for entity in entities])
                        
            whitelist_entities = set(one_node_short_answer_entities + multiple_node_short_answer_entities)
            
            # Write the whitelist to a file
            with gzip.open(whitelist_path, 'w') as f_out:
                for entity in whitelist_entities:
                    f_out.write(f"{entity}\n".encode('utf-8'))
                    
                    
def embed_nodes(data_dir, tokenizer, model, device, embedding_dim):    
    alias_file = 'wikidata5m_alias.tar.gz'
    alias_path = os.path.join(data_dir, alias_file)
    
    embeddings_file = f"wikidata5m_entity_embeddings_dim-{embedding_dim}.pkl"
    embeddings_path = os.path.join(data_dir, embeddings_file)

    # Check if embeddings are already computed
    if not os.path.exists(embeddings_path):
        # Load aliases
        embeddings = {}
        
        model.to(device)
        model.eval()
        
        with gzip.open(alias_path, 'rt', encoding='latin-1') as file:
            lines = file.readlines()[1:4813490]  # Adjust as needed
            for line in tqdm(lines, desc="Embedding Nodes", unit=" lines"):
                parts = line.strip().split('\t')
                entity_id, aliases = parts[0], parts[1:]

                # Compute and average the embeddings for all aliases of the entity
                inputs = tokenizer(aliases, padding=True, max_length=512, truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                avg_embedding = torch.mean(outputs.pooler_output.cpu(), dim=0).numpy()
                
                embeddings[entity_id] = avg_embedding

        # Save embeddings to disk
        with open(embeddings_path, 'wb') as f:
            joblib.dump(embeddings, f)


def build_faiss_index(data_dir, embedding_dim):
    """
    Build (if not exist) or load a FAISS index from the given file path.
    
    :return: The FAISS index and the node ID to index mapping.
    """
    embeddings_file = f"wikidata5m_entity_embeddings_dim-{embedding_dim}.pkl"
    embeddings_path = os.path.join(data_dir, embeddings_file)
    
    index_file = f"wikidata5m_node_dim-{embedding_dim}.faiss"
    index_path = os.path.join(data_dir, index_file)
    
    with open(embeddings_path, 'rb') as f:
        embedding_dict = joblib.load(f)
    
    if not os.path.exists(index_path) and os.path.exists(index_path + '.mapping'):
        # Convert embeddings to a matrix
        embedding_matrix = np.array(list(embedding_dict.values())).astype('float32')

        # Dimension of the embeddings
        d = embedding_matrix.shape[1]

        # Building the index
        index = faiss.IndexFlatL2(d)
        index.add(embedding_matrix)

        # Saving the index
        faiss.write_index(index, index_path)

        # Save a mapping of node ID to index in the matrix
        node_id_to_index = {node_id: i for i, node_id in enumerate(embedding_dict.keys())}
        with open(index_path + '.mapping', 'wb') as f:
            pickle.dump(node_id_to_index, f)


data_dir = os.path.join('data', 'wikidata5m')
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_dim = 128


embed_nodes(data_dir, tokenizer, model, device, embedding_dim)