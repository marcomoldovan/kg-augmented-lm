import io
import os
import re
import gzip
from typing import NamedTuple, List, Tuple, Iterator
import numpy as np

from src import utils

log = utils.get_pylogger(__name__)



class GraphTextPair(NamedTuple):
    """Text paired with raw graph represented as in `edges`."""
    center_node: str
    title: str
    edges: List[str]
    text: str
    
    
class WikitextArticle(NamedTuple):
    title: str
    text: str
    
    
class Graph(NamedTuple):
  title: str
  center: str
  edges: List[str]

    
    
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


def read_pairs_from_gzip_txt_file(file_path: str) -> Iterator[GraphTextPair]:
    """Read graph-text pairs from gzip txt files.

    Args:
        file_path: a `.gz` file of graph-text pairs written in the same format as
        using the `write_pairs_to_gzip_txt_file` function.

    Yields:
        Graph-text pairs from this file.
    """
    content = read_gzip_txt_file(file_path)

    graph_header_sep_re = re.compile(
        r'(<graph center=[^ ]+ title="[^"]+">)')
    graph_header_re = re.compile(
        r'<graph center=([^ ]+) title="([^"]+)">$')
    section_sep_re = re.compile(r'\n(<section id="[^"]+">\n)')
    parts = graph_header_sep_re.split(content)

    # Skip the first part which is empty
    for i in range(1, len(parts), 2):
        header, body = parts[i], parts[i + 1]
        m = graph_header_re.match(header)

        # 5 parts total, empty first part, "text", text section, "edges", edges
        # section.
        section_parts = section_sep_re.split(body)

        yield GraphTextPair(center_node=m.group(1),
                            title=m.group(2),
                            text=section_parts[2],
                            edges=section_parts[-1].strip().split('\n'))