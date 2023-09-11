"""Preprocess freebase data and pair with wikitext."""

import os
import io
import csv
import collections
from typing import List, Tuple


from src import utils
from src.data.components.wikigraphs.tokenizers import GraphTokenizer
from src.data.components.wikigraphs.dataset import RawDataset, ParsedDataset
import src.data.components.wikigraphs.utils as wikigraph_utils

log = utils.get_pylogger(__name__)


###############################################
############# Freebase Preprocessing ##########
###############################################

def pair_graphs_with_wikitext(subset: str, graph_dir: str, data_dir: str, output_dir: str):
    """Pair graphs with wikitext articles, and write to output directory."""
    if not os.path.exists(os.path.join(output_dir, f'{subset}.gz')):
        log.info('Pairing graphs from the %s set from %s with wikitext.',
                    subset, graph_dir)
        graphs = list(wikigraph_utils.graphs_from_file(
            os.path.join(graph_dir, f'{subset}.gz')))
        title2graph = {
            wikigraph_utils.normalize_freebase_string(g.title).replace(' ', ''): g
            for g in graphs}
        n_graphs = len(graphs)

        # Use raw version of the wikitext data as the tokenized version has <unk> in
        # titles which is bad for matching.  We will handle the <unk>s through the
        # tokenizer to make sure our data are equivalent to that of the tokenized
        # version of wikitext-103.
        wikitext_articles = list(RawDataset(subset=subset,data_dir=data_dir, version='raw'))
        n_wiki = len(wikitext_articles)
        log.info('Loaded %d graphs and %d wikitext articles in total.',
                    n_graphs, n_wiki)

        # Keep track of the article titles in the dataset.  Unfortunately wikitext-103
        # has about 1% of duplicated articles, we want to take care of that.
        retrieved_titles = set()
        pairs = []
        n_duplicates = 0
        for a in wikitext_articles:
            title = wikigraph_utils.normalize_title(a.title).replace(' ', '')
            g = title2graph.get(title, None)
            if g is not None:
                if title not in retrieved_titles:
                    retrieved_titles.add(title)
                    pairs.append(wikigraph_utils.GraphTextPair(
                        center_node=g.center,
                        title=g.title,
                        edges=g.edges,
                        text=a.text))
                else:
                    n_duplicates += 1

        n_pairs = len(pairs)
        log.info('Matched %d/%d = %.1f%% of wikitext articles,'
                    ' and %d/%d = %.1f%% of graphs.',
                    n_pairs, n_wiki, float(n_pairs) / n_wiki * 100,
                    n_pairs, n_graphs, float(n_pairs) / n_graphs * 100)
        log.info('Detected %d/%d = %.1f%% of duplicated wikitext articles.',
                    n_duplicates, n_wiki, float(n_duplicates) / n_wiki * 100)

        wikigraph_utils.write_pairs_to_gzip_txt_file(
            os.path.join(output_dir, f'{subset}.gz'), pairs)
    else:
        log.info('Skipping pairing graphs with wikitext: Files already exists at %s', output_dir)
    
    
def pair_graphs_with_wikitext_across_splits(data_dir, version):
    freebase_dir = os.path.join(data_dir, 'freebase', version)
    output_dir = os.path.join(data_dir, 'wikigraphs', version)
    for subset in ['train', 'valid', 'test']:
        pair_graphs_with_wikitext(subset, freebase_dir, data_dir, output_dir)


###############################################
############# Build Vocab #####################
###############################################

def get_vocab(dataset: RawDataset) -> List[Tuple[str, int]]:
    """Build vocabulary, return (word, count) tuples sorted by count."""
    vocab = collections.defaultdict(int)

    for pair in dataset:
        for t in pair.text.split(' '):
            if t:
                vocab[t] += 1

    return sorted(vocab.items(), key=lambda t: -t[1])


def write_vocab(vocab: List[Tuple[str, int]], output_path: str):
    """Write a vocab list to a file."""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, mode='wb') as f_:
        with io.TextIOWrapper(f_, encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerows(vocab)
      

def build_wikitext_vocab(data_dir):
    output_path = f'{data_dir}/wikitext-vocab.csv'
    if not os.path.isfile(output_path):
        log.info('Loading the dataset.')
        dataset = RawDataset(subset='train', data_dir=data_dir)
        log.info('Building the vocab.')
        vocab = get_vocab(dataset)
        log.info('Finished, vocab size %d, total number of tokens %d',
                    len(vocab), sum([c for _, c in vocab]))
        log.info('Writing the vocab to %s', data_dir)
        write_vocab(vocab, output_path)
    else:
        log.info('Skipping Wikitext vocab building: File already exists at %s', output_path)
    
    
def build_graph_vocab(data_dir, version, threshold):
    """Build vocabulary for graph data."""
    output_path = f'{data_dir}/graph-vocab.csv'
    if not os.path.isfile(output_path):
        log.info('Loading the dataset.')
        dataset = ParsedDataset(
            subset='train', data_dir=data_dir, version=version)
        log.info('Building graph vocab.')

        vocab = collections.defaultdict(int)
        for pair in dataset:
            graph = pair.graph
            for n in graph.nodes():
                for t in GraphTokenizer.split_node(n):
                    if t:
                        vocab[t] += 1
            for _, _, e in graph.edges():
                for t in GraphTokenizer.split_edge(e):
                    if t:
                        vocab[t] += 1

        vocab = sorted(vocab.items(), key=lambda t: -t[1])
        vocab = [k for k, v in vocab if v >= threshold]

        log.info('Finished, vocab size %d.', len(vocab))
        log.info('Writing the vocab to %s.', data_dir)

        wikigraph_utils.write_txt_file(output_path, '\n'.join(vocab),
                                # Some unicode characters requires utf-16 to encode.
                                encoding='utf-16')
    else:
        log.info('Skipping Graph vocab building: File already exists at %s', output_path)


def build_text_vocab(data_dir, version, threshold):
    """Build vocabulary for the text part of the graph-to-text data."""
    output_path = f'{data_dir}/text-vocab.csv'
    if not os.path.isfile(output_path):
        log.info('Loading the dataset.')
        dataset = ParsedDataset(
            subset='train', data_dir=data_dir, version=version)
        log.info('Building text vocab.')

        vocab = collections.defaultdict(int)
        for pair in dataset:
            for t in pair.text.split(' '):
                if t:
                    vocab[t] += 1

        vocab = sorted(vocab.items(), key=lambda t: -t[1])
        log.info('Finished, vocab size %d, total number of tokens %d.',
                    len(vocab), sum([v for _, v in vocab]))
        vocab = [(k, v) for k, v in vocab if v >= threshold]
        log.info('After filtering, vocab size %d.', len(vocab))
        log.info('Writing the vocab to %s.', data_dir)

        write_vocab(vocab, output_path)
    else:
        log.info('Skipping Text vocab building: File already exists at %s', output_path)