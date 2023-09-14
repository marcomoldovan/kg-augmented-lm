import re
import abc
import collections
from typing import Any, Iterator, Dict, List, Tuple, NamedTuple
import numpy as np
import torch
import torch.nn.functional as F

import src.data.components.wikigraphs.utils as utils
from src.data.components.wikigraphs.tokenizers import WordTokenizer, GraphTokenizer

class Graph:
    """A convenience class for representing graphs."""

    def __init__(self, nodes: List[str], edges: List[Tuple[int, int, str]]):
        """Construct a graph from a list of nodes and edges.

        Args:
        nodes: a list of node attributes, one for each node.
        edges: a list of (source_node_id, target_node_id, edge_attribute) for each
            edge.
        """
        self._nodes = nodes
        self._edges = edges
        self._node2id = {n: i for i, n in enumerate(nodes)}

    def nodes(self) -> List[str]:
        return self._nodes

    def edges(self) -> List[Tuple[int, int, str]]:
        return self._edges

    def node2id(self, node: str) -> int:
        return self._node2id[node]

    @classmethod
    def from_edges(cls, edges: List[str]) -> 'Graph':
        """Build a graph instance from a list of edges."""
        node2id = dict()
        parsed_edges = []
        next_node_id = 0

        for e in edges:
            src, edge, tgt = e.split('\t')[:3]
            src_id = node2id.get(src, next_node_id)
            if src_id == next_node_id:
                node2id[src] = src_id
                next_node_id += 1
            tgt_id = node2id.get(tgt, next_node_id)
            if tgt_id == next_node_id:
                node2id[tgt] = tgt_id
                next_node_id += 1
            parsed_edges.append((src_id, tgt_id, edge))

        id2node = {i: n for n, i in node2id.items()}
        return Graph(nodes=[id2node[i] for i in range(next_node_id)],
                    edges=parsed_edges)

    def to_edges(self) -> List[str]:
        r"""Convert graph to a list of edges.

        The converted list of edges should be compatible with the format specified
        in io_tools and compatible with the `from_edges` method above.

        Returns:
        edges: one edge per line, with the (source, target, edge_type) separated
            by `\t`.
        """
        edges = []
        for s, t, e in self._edges:
            edges.append(f'{self._nodes[s]}\t{e}\t{self._nodes[t]}')
        return edges

    @classmethod
    def subsample_nodes(
        cls, graph: 'Graph', subsample_rate: float = 1.0, center_node: str = None
        ) -> 'Graph':
        """Subsample the nodes of a graph."""
        graph_size = len(graph.nodes())
        if subsample_rate == 1.0 or graph_size <= 1:
            return graph
        subsampled_nodes_id = np.arange(graph_size)
        if subsample_rate < 1.0:
            subsample_graph_size = int(subsample_rate * graph_size)
            if center_node is not None:
                # We need to keep the center node during subsampling
                center_node_id = graph.node2id(center_node)
                subsampled_nodes_id = subsampled_nodes_id[
                    subsampled_nodes_id != center_node_id]
                subsample_graph_size = max(1, subsample_graph_size - 1)
                subsampled_nodes_id = np.random.choice(
                    subsampled_nodes_id, subsample_graph_size, replace=False)
                subsampled_nodes_id = np.append(subsampled_nodes_id, center_node_id)
            else:
                subsampled_nodes_id = np.random.choice(
                    subsampled_nodes_id, subsample_graph_size, replace=False)
            subsampled_nodes_id = np.sort(subsampled_nodes_id)
            map_subsampled_nodes_id = {
                old_id: new_id for new_id, old_id in enumerate(subsampled_nodes_id)}
        nodes = []
        edges = []
        for node_id, n in enumerate(graph.nodes()):
            if node_id in subsampled_nodes_id:
                nodes.append(n)
        for out_node, in_node, e in graph.edges():
            if out_node in subsampled_nodes_id and in_node in subsampled_nodes_id:
                edges.append((map_subsampled_nodes_id[out_node],
                            map_subsampled_nodes_id[in_node], e))
        return Graph(nodes=nodes, edges=edges)
    
    
class ParsedGraphTextPair(NamedTuple):
    """Graph-text pair with graph parsed into a `Graph` instance."""
    center_node: str
    title: str
    text: str
    graph: Graph
    
    
class GraphTensors(NamedTuple):
    """Graph representation as tensors can be fed directly into GAT."""
    nodes: torch.Tensor
    adj_mat: torch.Tensor
    

class SequenceChunk(NamedTuple):
    """A chunk of a sequence."""
    seq_idx: int
    start: int
    end: int
    

class Dataset(abc.ABC):
    """Base class for all datasets.

    All sub-classes should define `_load_data()` where an iterator
    `self._data_iter` should be instantiated that iterates over the dataset.
    """

    def __init__(self):
        """Constructor."""
        self._data_iter = None  # An iterator produced by `self._load_data`.

    @abc.abstractmethod
    def _load_data(self) -> Iterator[Any]:
        """Prepare data for another pass through the dataset.

        This method should return a generator in a child class.
        """

    def __next__(self):
        return next(self._data_iter)

    def __iter__(self):
        self._data_iter = self._load_data()
        return self


class RawDataset(Dataset):
    """Raw text dataset for wikitext-103."""

    def __init__(self,
                subset: str = 'train',
                shuffle_data: bool = False,
                data_dir: str = None,
                version: str = 'tokens'):
        """Constructor.

        Args:
        subset: which subset to load, one of {"train", "valid", "test"}.
        shuffle_data: if set to True the data will be randomly shuffled.
        data_dir: if provided will be used instead of the default `DATA_ROOT` as
            the directory that contains the data.
        version: one of {'tokens', 'raw'}
        """
        super().__init__()
        self._subset = subset
        self._shuffle_data = shuffle_data
        self._data_dir = data_dir
        self._dataset = None

        allowed_versions = ('tokens', 'raw')
        if version not in allowed_versions:
            raise ValueError(f'Version must be one of {allowed_versions}.')
        self._version = version

    def _load_data(self):
        """Prepare data for another pass through the dataset."""
        if self._dataset is None:
            data_root = self._data_dir + '/wikitext-103' + ('-raw' if self._version == 'raw' else '')
            self._dataset = utils.articles_from_file(
                f'{data_root}/wiki.{self._subset}.{self._version}')

        def source():
            n_articles = len(self._dataset)
            if self._shuffle_data:
                idx = np.random.permutation(n_articles)
            else:
                idx = np.arange(n_articles)
            for i in range(n_articles):
                yield self._dataset[idx[i]]
                
        return source()
    
    
class RawPairedDataset(Dataset):
    """The untokenized raw dataset."""

    def __init__(self,
                subset: str = 'train',
                shuffle_data: bool = False,
                data_dir: str = None,
                version: str = 'max256'):
        """Constructor.

        Args:
        subset: which subset to load.
        shuffle_data: set to True to randomly shuffle the data.
        data_dir: if provided this will be used instead of the default location to
            look for data, it must contain files like `train.gz`, `valid.gz` and
            `test.gz`.
        version: which version of the data to load, this must be the name of a
            directory in `DATA_ROOT`.
        """
        super().__init__()
        self._subset = subset
        self._shuffle_data = shuffle_data
        self._data_dir = data_dir
        self._dataset = None

        allowed_versions = ('max256', 'max512', 'max1024')
        if version not in allowed_versions:
            raise ValueError(f'Version {version} not one of the allowed versions:'
                            f' {allowed_versions}.')
        self._version = version

    def _load_data(self):
        """Load and prepare the data iterator."""
        if self._dataset is None:
            self._dataset = list(utils.read_pairs_from_gzip_txt_file(
                f'{self._data_dir}/wikigraphs/{self._version}/{self._subset}.gz'))

        def source():
            n_pairs = len(self._dataset)
            if self._shuffle_data:
                idx = np.random.permutation(n_pairs)
            else:
                idx = np.arange(n_pairs)
            for i in range(n_pairs):
                yield self._dataset[idx[i]]

        return source()
    
    
class ParsedDataset(Dataset):
    """Raw dataset + parsing graphs into Graph instances."""

    def __init__(self,
                subset: str = 'train',
                shuffle_data: bool = False,
                data_dir: str = None,
                version: str = 'max256'):
        """Constructor.

        Args:
        subset: which subset to load.
        shuffle_data: set to True to randomly shuffle the data.
        data_dir: if provided this will be used instead of the default location to
            look for data, it must contain files like `train.gz`, `valid.gz` and
            `test.gz`.
        version: which version of the data to load, this must be the name of a
            directory in `DATA_ROOT`.
        """
        super().__init__()
        self._raw_data = RawPairedDataset(subset=subset, shuffle_data=False,
                                    data_dir=data_dir, version=version)
        self._shuffle_data = shuffle_data
        self._dataset = None

    def _load_data(self):
        if self._dataset is None:
        # pylint: disable=g-complex-comprehension
            self._dataset = [ParsedGraphTextPair(center_node=pair.center_node,
                                                title=pair.title,
                                                text=pair.text,
                                                graph=Graph.from_edges(pair.edges))
                            for pair in self._raw_data]

        def source():
            n_pairs = len(self._dataset)
            if self._shuffle_data:
                idx = np.random.permutation(n_pairs)
            else:
                idx = np.arange(n_pairs)
            for i in range(n_pairs):
                yield self._dataset[idx[i]]

        return source()
        
    
    
class WikigraphDataset(torch.utils.data.Dataset):
    def __init__(self,
                 tokenizer: WordTokenizer,
                 graph_tokenizer: GraphTokenizer,
                 data_dir: str = None,
                 text_only: bool = False,
                 batch_size: int = 1,
                 timesteps: int = 128,
                 subsample_rate: float = 1.0,
                 version: str = 'max256',
                 subset: str = 'train',
                 max_center_split_offset: int = 16,
                 split_overlap: int = 1
                 ) -> None:
        super().__init__()
                
        self._tokenizer = tokenizer
        self._graph_tokenizer = graph_tokenizer
        self._data_dir = data_dir
        self._text_only = text_only
        self._batch_size = batch_size
        self._timesteps = timesteps
        self._graph_feature_dim = graph_tokenizer.vocab_size
        self._pad_value = self._tokenizer.pad_token()
        self._subsample_nodes = subsample_rate
        self._version = version 
        self._subset = subset
        self._sentence_delimiters = self._tokenizer.sentence_delimiters()
        self._max_center_split_offset = max_center_split_offset
        self._split_overlap = split_overlap
        
        allowed_versions = ('max256', 'max512', 'max1024')
        if version not in allowed_versions:
            raise ValueError(f'Version {version} not one of the allowed versions:'
                            f' {allowed_versions}.')
            
        # 1. RawDataset
        self._raw_data = list(utils.read_pairs_from_gzip_txt_file(
          f'{self._data_dir}/wikigraphs/{self._version}/{self._subset}.gz'))
        
        # 2. ParsedDataset
        self._parsed_data = [ParsedGraphTextPair(center_node=pair.center_node,
                                                 title=pair.title,
                                                 text=pair.text,
                                                 graph=Graph.from_edges(pair.edges))
                             for pair in self._raw_data]
        
        # 3. BaseGraph2TextDataset
        self._dataset = [self._process_graph_text_pair(p) for p in self._parsed_data]
        
        # 4. Graph tensors
        self._graph_tensors = [self._raw_graph_to_tensor(*graph) for graph, _ in self._dataset]
        
        # 5. Chunked Dataset
        self._chunked_sequences = [chunk for idx, (_, seq) in enumerate(self._dataset) 
                                   for chunk in self._chunk_graph_text_pair(idx, seq)]
        
        # 6. Count total number of tokens
        self._n_tokens = sum([len(seq) for _, seq in self._dataset])
      
      
    def _process_graph(self, center_node: str, graph: Graph):
        """Process the graph part of a `ParsedGraphTextPair` instance."""
        
        if self._subsample_nodes < 1.0:
            graph = Graph.subsample_nodes(graph, self._subsample_nodes, center_node)

        nodes = graph.nodes()
        edges = graph.edges()
        n_edges = len(edges)

        sender = np.zeros(n_edges, dtype=np.int32)
        receiver = np.zeros(n_edges, dtype=np.int32)

        nodes_bow = []
        edges_bow = []

        for n in nodes:
            bow = collections.defaultdict(int)
            for t in self._graph_tokenizer.encode_node(n):
                bow[t] += 1
            nodes_bow.append(bow)
        for i, (s, r, e) in enumerate(edges):
            bow = collections.defaultdict(int)
            for t in self._graph_tokenizer.encode_edge(e):
                bow[t] += 1
            edges_bow.append(bow)
            sender[i] = s
            receiver[i] = r

        return (nodes_bow, edges_bow, sender, receiver, graph.node2id(center_node))


    def _process_graph_text_pair(
        self, pair: ParsedGraphTextPair) -> Tuple[Any, np.ndarray]:
        """Process the given graph-text pair and prepare one example.

        Args:
        pair: the input `ParsedGraphTextPair` instance.

        Returns:
        graph: the processed graph content.
        text: the tokenized text, a sequence of token IDs.
        """
        return (self._process_graph(pair.center_node, pair.graph),
                self._tokenizer.encode(
                    pair.text, prepend_bos=True, append_eos=True))
    
    
    def _raw_graph_to_tensor(
        self, nodes_bow, edges_bow, sender, receiver, center_node_id
        ) -> GraphTensors:
        """Convert the input to a named tuple instance containing pytorch 
        tensors for nodes and adjacency matrix.
        """
        n_nodes = len(nodes_bow)
        n_edges = len(edges_bow)

        # +1 for the center node indicator
        nodes = np.zeros((n_nodes, self._graph_feature_dim + 1), dtype=np.float32)
        edges = np.zeros((n_edges, self._graph_feature_dim), dtype=np.float32)

        nodes[center_node_id][-1] = 1
        for i, bow in enumerate(nodes_bow):
            for t, c in bow.items():
                nodes[i][t] = c
        for i, bow in enumerate(edges_bow):
            for t, c in bow.items():
                edges[i][t] = c

        # Convert nodes to PyTorch tensors
        nodes_tensor = torch.tensor([np.argmax(row) for row in nodes], dtype=torch.float32)
        
        # Create adjacency matrix
        adjacency_matrix = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
        for i in range(n_edges):
            adjacency_matrix[sender[i], receiver[i]] = np.argmax(edges[i]).item()

        return GraphTensors(nodes=nodes_tensor, adj_mat=adjacency_matrix)
        
        
    def _chunk_graph_text_pair(self, seq_idx, seq):
        """Compute valid chunks for a single sequence.""" 
        chunks = []
        position = 0
        while position < len(seq):
            chunk_end = position + self._timesteps
            if chunk_end > len(seq):
                # If this is the last chunk of the sequence and it's shorter than timesteps
                chunk_end = len(seq)
            else:
                # Adjust the chunk to end on the last sentence end index before chunk_end
                adjusted_end = max([i for i in range(position, chunk_end) 
                                    if seq[i] in self._sentence_delimiters], default=chunk_end-1) + 1
                chunk_end = adjusted_end

            seq_chunk = SequenceChunk(seq_idx, position, chunk_end)
            chunks.append(seq_chunk)
            position = chunk_end
        
        return chunks
    
    
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
        return len(self._chunked_sequences)
    
    
    def __getitem__(self, index) -> Any:
        chunk = self._chunked_sequences[index]
        seq = self._dataset[chunk.seq_idx][1][chunk.start:chunk.end]
        input_seq, target_seq = self._split_seq_into_input_target(seq)
        if self._text_only:
            return dict(input_seq=input_seq, target_seq=target_seq)
        else:
            graph = self._graph_tensors[chunk.seq_idx]
            return dict(graph=graph, input_seq=input_seq, target_seq=target_seq)


    def collate_fn(self, batch): #TODO add text only functionality
        max_seq_len = max(
            [len(e['input_seq']) for e in batch] + [len(e['target_seq']) for e in batch])
        max_nodes = max([e['graph'].nodes.size(0) for e in batch])
        max_adj_mat_size = max([e['graph'].adj_mat.size(0) for e in batch])
        
        input_attention_masks = []
        target_attention_masks = []
        nodes = []
        adj_mats = []
        
        for e in batch:
            input_attn_mask = torch.zeros(max_seq_len, dtype=torch.float32)
            input_attn_mask[:len(e['input_seq'])] = 1
            input_attention_masks.append(input_attn_mask)
            e['input_seq'] = F.pad(
                torch.tensor(e['input_seq']),
                (0, max_seq_len - e['input_seq'].shape[-1]), 
                'constant', 
                self._pad_value)
            target_attention_mask = torch.zeros(max_seq_len, dtype=torch.float32)
            target_attention_mask[:len(e['target_seq'])] = 1
            target_attention_masks.append(target_attention_mask)
            e['target_seq'] = F.pad(
                torch.tensor(e['target_seq']),
                (0, max_seq_len - e['target_seq'].shape[-1]), 
                'constant', 
                self._pad_value)
            nodes.append(F.pad(
                e['graph'].nodes,
                (0, max_nodes - e['graph'].nodes.shape[-1]),
                'constant', 
                self._pad_value))
            adj_mat = e['graph'].adj_mat
            adj_mats.append(F.pad(
                adj_mat, 
                (0, max(0, max_adj_mat_size - adj_mat.size(1)), 0, max(0, max_adj_mat_size - adj_mat.size(0))),
                'constant',
                value=self._pad_value)[:max_adj_mat_size, :max_adj_mat_size])
            
        input_seq = torch.stack([e['input_seq'] for e in batch], dim=0)
        input_attention_masks = torch.stack(input_attention_masks, dim=0)
        target_seq = torch.stack([e['target_seq'] for e in batch], dim=0)
        target_attention_masks = torch.stack(target_attention_masks, dim=0)
        node_features = torch.stack(nodes, dim=0)
        adj_mats = torch.stack(adj_mats, dim=0)
        
        return dict(
            input_seq=input_seq,
            input_attention_masks=input_attention_masks,
            target_seq=target_seq,
            target_attention_masks=target_attention_masks, 
            node_features=node_features, 
            adj_mats=adj_mats)
        
  
    def return_faux_batch(self) -> Dict[str, np.ndarray]:
        """Return a fake batch with the right shapes and dimensions."""
        input_seq = torch.randint(
            low=0, high=self._tokenizer.vocab_size, size=(self._batch_size, self._timesteps))
        input_attention_masks = torch.ones(
            (self._batch_size, self._timesteps), dtype=torch.float32)
        target_seq = torch.randint(
            low=0, high=self._tokenizer.vocab_size, size=(self._batch_size, self._timesteps))
        target_attention_masks = torch.ones(
            (self._batch_size, self._timesteps), dtype=torch.float32)
        node_features = torch.randint(
            low=0, high=self._graph_feature_dim, size=(self._batch_size, self._timesteps))
        adj_mats = torch.randint(
            low=0, high=self._graph_feature_dim, size=(self._batch_size, self._timesteps, self._timesteps))

        return dict(
            input_seq=input_seq,
            input_attention_masks=input_attention_masks,
            target_seq=target_seq,
            target_attention_masks=target_attention_masks, 
            node_features=node_features, 
            adj_mats=adj_mats)
        