import io
import os
import re
import gzip
from typing import NamedTuple, List, Tuple, Iterator
import numpy as np

from src import utils

log = utils.get_pylogger(__name__)

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


class GraphTextPair(NamedTuple):
    """Text paired with raw graph represented as in `edges`."""
    center_node: str
    title: str
    edges: List[str]
    text: str
    
    
class WikitextArticle(NamedTuple):
    title: str
    text: str
    
    
def read_gzip_txt_file(file_path: str, encoding: str = 'utf-8') -> str:
    """Read gzipped txt file."""
    with open(file_path, 'rb') as f:
        content = f.read()

    with gzip.GzipFile(fileobj=io.BytesIO(content), mode='rb') as f:
        content = f.read()
    return content.decode(encoding) 


def read_txt_file(file_path: str, encoding: str = 'utf-8') -> str:
    """Read a plain txt file."""
    with open(file_path, 'rb') as f:
        content = f.read()
    return content.decode(encoding)


def write_txt_file(file_path: str, txt: str, encoding: str = 'utf-8'):
    """Write the given txt string to file."""
    make_dir_if_necessary(file_path)
    with open(file_path, 'wb') as f:
        f.write(txt.encode(encoding, 'surrogatepass'))


def articles_from_file(file_path: str) -> List[WikitextArticle]:
    """Read wikitext articles from file.

    Args:
        file_path: path to the input `.tokens` file.

    Returns:
        A list of `WikitextArticle` tuples.
    """
    with open(file_path, mode='rb') as f:
        content = f.read()
    content = content.decode('utf-8')

    title_re = re.compile(r'(\n = ([^=].*) = \n \n)')
    parts = title_re.split(content)

    # Skip the first part which is empty
    return [WikitextArticle(title=parts[i+1], text=parts[i] + parts[i+2])
            for i in range(1, len(parts), 3)]


def graphs_from_file(file_path: str) -> Iterator[Graph]:
    """Read freebase graphs from file.

    Args:
        file_path: path to the input `.gz` file that contains a list of graphs.

    Yields:
        graphs: a list of read from the file.
    """
    content = read_gzip_txt_file(file_path)

    graph_header_sep_re = re.compile(
        r'(<graph center=[^ ]+ title="[^"]+">\n)')
    graph_header_re = re.compile(
        r'<graph center=([^ ]+) title="([^"]+)">\n')
    parts = graph_header_sep_re.split(content)

    # Skip the first part which is empty
    for i in range(1, len(parts), 2):
        header, body = parts[i], parts[i + 1]
        m = graph_header_re.match(header)
        yield Graph(title=m.group(2),
                    center=m.group(1),
                    edges=body.strip().split('\n'))


def normalize_freebase_string(s: str) -> str:
    """Expand the `$xxxx` escaped unicode characters in the input string."""
    # '"' is escaped as '``', convert it back.
    _UNICODE_RE = re.compile(r'(\$[0-9A-Fa-f]{4})')
    s.replace('``', '"')
    parts = _UNICODE_RE.split(s)
    parts = [p if not _UNICODE_RE.match(p) else chr(int(p[1:], base=16))
            for p in parts]
    return ''.join(parts).replace('_', ' ')


def normalize_title(title: str) -> str:
    """Normalize the wikitext article title by handling special characters."""
    return title.replace(
        '@-@', '-').replace('@,@', ',').replace('@.@', '.').replace(' ', '')
    
    
def pair2lines(pair):
    lines = [f'<graph center={pair.center_node} title="{pair.title}">']
    lines.append('<section id="text">')
    lines.append(pair.text)
    lines.append('<section id="edges">')
    lines.extend(pair.edges)
    return lines


def make_dir_if_necessary(output_path):
  output_dir = os.path.dirname(output_path)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def write_lines_to_gzipped_file(file_path, lines):
    make_dir_if_necessary(file_path)
    with open(file_path, 'wb') as f_zip:
        with gzip.GzipFile(fileobj=f_zip, mode='wb') as f:
            f.write('\n'.join(lines).encode('utf-8'))
    
    
def write_pairs_to_gzip_txt_file(file_path, pairs):
    log.info('Writing %d pairs to %s.', len(pairs), file_path)
    lines = []
    for p in pairs:
        lines.extend(pair2lines(p))
    write_lines_to_gzipped_file(file_path, lines)