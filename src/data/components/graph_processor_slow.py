import os
import gzip
import multiprocessing
from functools import lru_cache
import igraph as ig
from igraph import Graph


class GraphProcessor:
    def __init__(self, triples_path, save_dir, parallel=True):        
        # Load graph from disk if it exists, otherwise create it and save it
        if save_dir is None:
            self.graph = self.build_graph(triples_path, parallel)
        elif save_dir is not None and os.path.exists(os.path.join(save_dir, "full_graph.pickle")):
            self.graph = ig.Graph.Read_Pickle(os.path.join(save_dir, "full_graph.pickle"))
        elif save_dir is not None and  not os.path.exists(os.path.join(save_dir, "full_graph.pickle")):
            self.graph = self.build_graph(triples_path, parallel)
            self.graph.write_pickle(os.path.join(save_dir, "full_graph.pickle"))
                
    def build_graph(self, triples_path, use_multiprocessing):
        """Build a graph from the provided triples file."""
        if use_multiprocessing:
            return self.build_graph_from_triples_parallel(triples_path)
        else:
            return self.build_graph_from_triples(triples_path)

    def build_graph_from_triples(self, tar_gz_path):
        """Build a graph from the provided triples file compressed as tar.gz."""

        triples = []
        
        with gzip.open(tar_gz_path, 'rt') as file:
            lines = file.readlines()
            triples.extend([tuple(line.strip().split('\t')) for line in lines[1:]])
            
        triples = [rel for rel in triples if len(rel) == 3]

        # Extracting unique entities and relations from triples
        entities = list(set([s for s, _, _ in triples] + [t for _, _, t in triples]))

        # Create an empty directed graph
        g = ig.Graph(directed=True)

        # Add vertices to the graph
        g.add_vertices(entities)

        # Add edges to the graph
        for s, r, t in triples:
            g.add_edge(s, t, name=r)

        return g
    
    def build_subgraph(self, triples):
        """Construct a subgraph from a chunk of triples."""
        g = Graph(directed=True)

        # Extracting unique entities from triples
        entities = list(set([s for s, _, _ in triples] + [t for _, _, t in triples]))
        g.add_vertices(entities)

        # Prepare a list of edge tuples and their corresponding relations
        edge_list = [(s, t) for s, _, t in triples if s in entities and t in entities]
        relations = [r for s, r, t in triples if (s, t) in edge_list]

        # Add edges to the graph
        g.add_edges(edge_list)

        # Assign relation types as attributes to the edges
        g.es['relation'] = relations

        return g
    
    def merge_subgraphs(self, subgraphs):
        """Merges a list of subgraphs into a single graph."""
        main_graph = Graph(directed=True)
        
        for subgraph in subgraphs:
            main_graph.add_vertices(subgraph.vs["name"])  # Add vertices by their names
            main_graph.add_edges([(v.source_vertex["name"], v.target_vertex["name"]) for v in subgraph.es])  # Add edges by source and target vertex names

        return main_graph


    def build_graph_from_triples_parallel(self, tar_gz_path):
        """Build a graph from the provided triples file using parallel processing."""

        # Extract triples using the gzip approach
        with gzip.open(tar_gz_path, 'rt') as file:
            lines = file.readlines()
            triples = [tuple(line.strip().split('\t')) for line in lines[1:]]

        # Split the triples into chunks
        chunk_size = len(triples) // multiprocessing.cpu_count()
        chunks = [triples[i:i+chunk_size] for i in range(0, len(triples), chunk_size)]

        # Use a multiprocessing pool to construct subgraphs in parallel
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            subgraphs = pool.map(self.build_subgraph, chunks)

        # Merge the subgraphs into a single graph
        main_graph = self.merge_subgraphs(subgraphs)

        return main_graph
    

    @lru_cache(maxsize=None) # unlimited cache size
    def get_k_hop_neighbors(self, node, k=2):
        """Retrieve k-hop neighbors for a given node."""
        neighbors = set()
        for hop in range(1, k+1):
            neighbors.update(self.graph.neighborhood(node, order=hop))
        return list(neighbors)

    def prune_by_degree(self, subgraph, k=2, max_nodes=256):
        """Prune the subgraph based on node degree."""
        while len(subgraph.vs) > max_nodes:
            # Start with nodes at the outer boundary (i.e., k-hops away)
            to_delete_ids = [v.index for v in subgraph.vs if subgraph.degree(v) <= k]
            if not to_delete_ids:  # If we can't prune any more distant nodes, prune closer ones
                k -= 1
            if k == 0:  # If we can't prune based on degree, prune randomly
                to_delete_ids = [v.index for v in subgraph.vs][:len(subgraph.vs) - max_nodes]
            subgraph.delete_vertices(to_delete_ids)
        return subgraph

    def get_subgraph_for_entities(self, entities, k=2, max_nodes=256):
        """Retrieve a subgraph for a given list of entities."""
        all_neighbors = set()
        for entity in entities:
            all_neighbors.update(self.get_k_hop_neighbors(entity, k))
        subgraph = self.graph.subgraph(list(all_neighbors))
        pruned_subgraph = self.prune_by_degree(subgraph, k=k, max_nodes=max_nodes)
        return pruned_subgraph
    
    
from time import time

start_time = time()
gp = GraphProcessor('./data/wikidata5m/wikidata5m_transductive_100k.tar.gz', None, True)
end_time = time()
print("Time taken for igraph on 100k triples with multiprocessing: {:.2f} seconds".format(end_time - start_time))

start_time = time()
gp = GraphProcessor('./data/wikidata5m/wikidata5m_transductive_1m.tar.gz', None, True)
end_time = time()
print("Time taken for igraph on 1m triples with multiprocessing: {:.2f} seconds".format(end_time - start_time))
