import re
import abc
import collections
from typing import Any, Iterator, Dict, List, Tuple, NamedTuple
import numpy as np
import torch

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
                 ) -> None:
        super().__init__()
                
        self._tokenizer = tokenizer
        self._graph_tokenizer = graph_tokenizer
        self._data_dir = data_dir
        self._text_only = text_only #TODO implement text-only option
        self._batch_size = batch_size
        self._timesteps = timesteps
        self._graph_feature_dim = None
        self._pad_value = None
        self._placeholder_graph = None
        self._subsample_nodes = subsample_rate
        self._version = version 
        self._subset = subset
        
        allowed_versions = ('max256', 'max512', 'max1024')
        if version not in allowed_versions:
            raise ValueError(f'Version {version} not one of the allowed versions:'
                            f' {allowed_versions}.')
            
        # 1. RawDataset
        self._raw_data = list(utils.read_pairs_from_gzip_txt_file(
          f'{self._data_dir}/{self._version}/{self._subset}.gz'))
        
        # 2. ParsedDataset
        self._parsed_data = [ParsedGraphTextPair(center_node=pair.center_node,
                                                 title=pair.title,
                                                 text=pair.text,
                                                 graph=Graph.from_edges(pair.edges))
                             for pair in self._raw_data]
        
        # 3. BaseGraph2TextDataset
        self._dataset = [self._process_graph_text_pair(p) for p in self._parsed_data]
      
      
    def __len__(self):
        return len(self._dataset)
    
    
    def __getitem__(self, index) -> Any:
        (graph, seq), (graph_id, seq_id) = self._dataset[index]
        return dict(obs=seq[:self._timesteps], graph=graph, seq_id=seq_id, graph_id=graph_id)
    
    
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
                 
        
    def _to_graph_with_features(self, nodes_bow, edges_bow, sender, receiver, center_node_id):
        """Convert the input to a more readable dict"""
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
                
        return dict(nodes=nodes, edges=edges, senders=sender, receivers=receiver,
            globals=None, n_node=np.array([n_nodes], dtype=np.int32),
            n_edge=np.array([n_edges], dtype=np.int32))
    
    
    def _process_graph_batch(self, graphs: List[Any]):
        """Process a batch of graph data.

        Args:
        graphs: a list of graph data, each as returned by `_process_graph`.

        Returns:
        processed_graphs: a list of processed tensor(s).
        """
        graphs = [g if g is not None else self._placeholder_graph for g in graphs]
        return [self._to_graph_with_features(*g) for g in graphs]

    
    def _pad_to(self, x: np.array, size: int, axis: int = -1, pad_value: float = 0.):
        """Pad an array to the specified size along a specified axis."""
        if x.shape[axis] > size:
            raise ValueError(f'Data item has size {x.shape[axis]} larger than {size}'
                            f' in axis {axis} already.')
        elif x.shape[axis] == size:
            return x
        else:
            pad_amount = [(0, 0)] * x.ndim
            pad_amount[axis] = (0, size - x.shape[axis])
            return np.pad(x, pad_amount, mode='constant', constant_values=pad_value)


    def collate_fn(self, batch):
        for e in batch:
            e['obs'] = self._pad_to(e['obs'], self._timesteps, axis=0, pad_value=self._pad_value)
        batch = dict(
            obs=np.stack([e['obs'] for e in batch], axis=0),
            graphs=[e['graph'] for e in batch],
            should_reset=np.stack([e['should_reset'] for e in batch], axis=0))
        batch = map(lambda x: dict(  # pylint: disable=g-long-lambda #??????
            obs=x['obs'][:, :-1],
            target=x['obs'][:, 1:],
            should_reset=x['should_reset'][:, :-1],
            # If target is a <pad> token then that target should not be predicted.
            mask=(x['obs'][:, 1:] != self._tokenizer.pad_token()).astype(
                np.float32),
            graphs=self._process_graph_batch(x['graphs']),
            ), batch)
        return batch
        
  
    def return_faux_batch(self) -> Dict[str, np.ndarray]:
        """Return a fake batch with the right shapes and dimensions."""
        obs = torch.zeros([self._batch_size, self._timesteps], dtype=torch.int32)
        target = torch.zeros([self._batch_size, self._timesteps], dtype=torch.int32)
        should_reset = torch.zeros_like(obs, dtype=torch.float32)
        mask = torch.zeros_like(obs, dtype=torch.float32)
        
        # A batch should contain `batch_size` graphs.
        # Here we make sure each graph has one node and one edge.
        graphs = []
        for _ in range(self._batch_size):
            nodes = torch.zeros([1, self._graph_feature_dim + 1], dtype=torch.float32)
            edges = torch.zeros([1, self._graph_feature_dim], dtype=torch.float32)
            senders = torch.zeros([1], dtype=torch.int32)
            receivers = torch.zeros([1], dtype=torch.int32)
            n_node = torch.ones(1, dtype=torch.int32)
            n_edge = torch.ones(1, dtype=torch.int32)
            globals_ = None  # Change this if you have global features

            graphs.append({
                'nodes': nodes,
                'edges': edges,
                'senders': senders,
                'receivers': receivers,
                'n_node': n_node,
                'n_edge': n_edge,
                'globals': globals_
            })

        return {
            'obs': obs,
            'target': target,
            'mask': mask,
            'should_reset': should_reset,
            'graphs': graphs
        }
        