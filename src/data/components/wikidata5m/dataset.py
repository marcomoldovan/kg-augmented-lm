import gzip

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.data.components.entity_linker import EntityLinker
from src.data.components.graph_processor import GraphProcessor

class Wikidata5MDataset(Dataset):
    def __init__(self, corpus_path, entity_linker: EntityLinker, graph_processor: GraphProcessor, tokenizer, k_hop: int = 2, max_nodes: int = 256, timesteps: int = 128, chunk_documents: bool = True, split_into_src_tgt: bool = True):
        self.el = entity_linker
        self.gp = graph_processor
        self.tok = tokenizer
        
        self.k_hop = k_hop
        self.max_nodes = max_nodes
        self.timesteps = timesteps
        self.chunk_documents = chunk_documents
        self.split_into_src_tgt = split_into_src_tgt
        
        with gzip.open(corpus_path, 'rt') as f:
            self.corpus = f.readlines()
        
    def _split_seq_into_input_target(self, seq):
        delimiter_indices = [i for i, id in enumerate(seq) if id in self._sentence_delimiters]
        if (len(seq) % 2) == 0:
            center_index = (len(seq) // 2) - 1
        else:
            center_index = len(seq) // 2

        # If there are no delimiters, split the list in the middle
        if not delimiter_indices:
            return seq[:center_index + self._split_overlap], seq[center_index:]

        # Find the delimiter index closest to the center of the list
        closest_delimiter_index = min(delimiter_indices, key=lambda i: abs(i - center_index))
        if abs(center_index - closest_delimiter_index) > self._max_center_split_offset:
            return seq[:center_index + self._split_overlap], seq[center_index:]

        # If the delimiter is at the end of the list, split the list in the middle, excluding the delimiter
        if closest_delimiter_index == len(seq) - 1:
            return seq[:center_index + self._split_overlap], seq[center_index:]

        # Otherwise, split the list at the delimiter index
        return seq[:closest_delimiter_index + self._split_overlap + 1], seq[closest_delimiter_index + 1:]
        
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        _, text = self.corpus[idx].strip().split('\t')
        entities = self.el.link_entities(text)
        subgraph = self.gp.get_subgraph_for_entities(entities, self.k_hop, self.max_nodes)
        
        # if self.split_text
        
    def collate_fn(self, batch):
        pass