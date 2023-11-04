import os
import gzip
from tqdm import tqdm
from graph_tool import Graph, GraphView, load_graph
from graph_tool.util import find_vertex


class GraphProcessor:
    def __init__(
        self, 
        data_dir, 
        triples_file='wikidata5m_transductive.tar.gz', 
        graph_file='wikidata5m_graph.gt.gz', 
        max_subgraph_edges=1000000):
        
        triples_path = os.path.join(data_dir, triples_file)
        graph_path = os.path.join(data_dir, graph_file)
        
        self.graph = self.build_full_graph(data_dir, triples_path, graph_path, max_subgraph_edges)
        
        print("Number of entities: {}".format(self.graph.num_vertices()))
        print("Number of edge types: {}".format(self.num_edge_types(self.graph)))
        print("Number of relations: {}".format(self.graph.num_edges()))

    def build_graph(self, triples):
        g = Graph(directed=True)
        vprop = g.new_vertex_property("string")
        eprop = g.new_edge_property("string")

        vertices = {}
        for s, p, o in triples:
            if s not in vertices:
                v1 = g.add_vertex()
                vprop[v1] = s
                vertices[s] = v1
            else:
                v1 = vertices[s]

            if o not in vertices:
                v2 = g.add_vertex()
                vprop[v2] = o
                vertices[o] = v2
            else:
                v2 = vertices[o]

            e = g.add_edge(v1, v2)
            eprop[e] = p

        g.vertex_properties["name"] = vprop
        g.edge_properties["name"] = eprop

        return g
    
    def merge_subgraph(main_graph, subgraph):
        # Dictionary to store the mapping of old vertices in subgraph to new vertices in main_graph
        vertex_map = {}

        for v in subgraph.vertices():
            v_name = subgraph.vp.name[v]
            
            # Check if vertex already exists in main_graph
            existing_vertices = find_vertex(main_graph, main_graph.vp.name, v_name)
            if existing_vertices:
                v_new = existing_vertices[0]
            else:
                v_new = main_graph.add_vertex()
                main_graph.vp.name[v_new] = v_name

            vertex_map[v] = v_new

        for e in subgraph.edges():
            source = vertex_map[e.source()]
            target = vertex_map[e.target()]
            e_new = main_graph.add_edge(source, target)
            main_graph.ep.name[e_new] = subgraph.ep.name[e]

    
    def build_full_graph(self, data_dir, triples_path, graph_path, max_subgraph_edges=1000000):       
        if os.path.exists(graph_path):
            with gzip.open(graph_path, 'rb') as f:
                return load_graph(f)
            
        else:
            # Extract triples using the gzip approach
            with gzip.open(triples_path, 'rt') as file:
                lines = file.readlines()[1:-1]
            
            num_chunks = -(-len(lines) // max_subgraph_edges)  # Ceiling division
            subgraphs = []  # to store paths of saved subgraphs
                    
            for i in tqdm(range(num_chunks), desc="Building Subgraphs"):
                start = i * max_subgraph_edges
                end = min((i + 1) * max_subgraph_edges, len(lines))
                triples_chunk = [tuple(line.strip().split('\t')) for line in lines[start:end]]
                
                subgraph = self.build_graph(triples_chunk)
                
                subgraph_path = os.path.join(data_dir, f"wikidata5m_subgraph_{i}.gt")
                subgraph.save(subgraph_path)
                subgraphs.append(subgraph_path)

            # Start with the first subgraph
            main_graph = load_graph(subgraphs[0])
            
            # Add the other subgraphs to it one by one
            for subgraph_path in tqdm(subgraphs[1:], desc="Merging Subgraphs"):
                subgraph = load_graph(subgraph_path)
                self.merge_subgraph(main_graph, subgraph)
                
            # Saving the final merged graph in gzipped format
            with gzip.open(graph_path, 'wb') as f:
                main_graph.save(f)
            
            # Deleting the subgraph files
            for subgraph_path in subgraphs:
                os.remove(subgraph_path)

            return main_graph
        
        
    def num_edge_types(graph, edge_property_name="name"):
        # Extract the edge property values into a set to get unique values
        unique_edge_types = set(graph.ep[edge_property_name])
        return len(unique_edge_types)
    
    
    def find_edge(graph, source_name, target_name):
        """
        Given a graph, a source entity name, and a target entity name,
        this function checks if there's an edge between the two entities.
        If the edge exists, it returns the relation type (edge property).
        Otherwise, it returns None.
        """
        vprop = graph.vp.name
        eprop = graph.ep.name
        
        source_vertex = find_vertex(graph, vprop, source_name)
        target_vertex = find_vertex(graph, vprop, target_name)

        if not source_vertex or not target_vertex:
            return None

        # Since find_vertex returns a list, we get the first item as our vertex.
        source_vertex = source_vertex[0]
        target_vertex = target_vertex[0]

        edge = graph.edge(source_vertex, target_vertex)

        if edge:
            return eprop[edge]
        else:
            return None


    def get_k_hop_neighbors(self, node, k=2):
        """Retrieve k-hop neighbors for a given node."""
        v = find_vertex(self.graph, self.vprop, node)
        if v:
            v = v[0]
            neighbors = set()
            for e in self.graph.edges():
                if e.source() == v or e.target() == v:
                    neighbors.add(int(self.vprop[e.source()]))
                    neighbors.add(int(self.vprop[e.target()]))
            return list(neighbors)
        else:
            return []

    def prune_by_degree(self, vertices, k=2, max_nodes=256):
        """Prune the subgraph based on node degree."""
        while len(vertices) > max_nodes:
            degree_dict = {v: self.graph.vertex(v).out_degree() for v in vertices}
            sorted_vertices = sorted(degree_dict.keys(), key=lambda x: degree_dict[x])
            vertices = sorted_vertices[:max_nodes]
        return vertices

    def get_subgraph_for_entities(self, entities, k=2, max_nodes=256):
        """Retrieve a subgraph for a given list of entities."""
        all_neighbors = set()
        for entity in entities:
            all_neighbors.update(self.get_k_hop_neighbors(entity, k))
        subgraph_vertices = list(all_neighbors)
        pruned_subgraph_vertices = self.prune_by_degree(subgraph_vertices, k=k, max_nodes=max_nodes)

        subgraph = GraphView(self.graph, vfilt=lambda v: self.vprop[v] in pruned_subgraph_vertices)

        return subgraph


from time import time

print(os.getcwd())

gp = GraphProcessor('./data/wikidata5m/')
entities = [('Q76', 0, 11), ('Q782', 25, 30), ('Q61061', 61, 69), ('Q30', 78, 90), ('Q8445', 111, 117), ('Q200', 145, 147), ('Q444353', 171, 175), ('Q61061', 183, 191), ('Q30', 200, 212), ('Q3220821', 230, 236), ('Q30', 245, 257), ('Q30', 262, 268), ('Q61', 273, 287), ('Q11651', 293, 299), ('Q61061', 301, 309), ('Q30', 318, 330), ('Q30', 335, 341), ('Q2057908', 346, 351), ('Q22686', 360, 371), ('Q22686', 390, 401), ('Q8445', 406, 412), ('Q16279311', 417, 423), ('Q22686', 432, 443), ('Q203', 449, 452), ('Q22686', 464, 475), ('Q61061', 489, 497), ('Q30', 506, 518)]
entities = [entity[0] for entity in entities]
print(len(entities))

start_time = time()
subgraph = gp.get_subgraph_for_entities(entities)
end_time = time()
print("Time taken to retrieve subgraph: {:.2f} seconds".format(end_time - start_time))