import torch.nn.functional as F


def masked_prediction_loss(node_logits, true_nodes, node_masks, adj_logits, true_adj, adj_masks):
    """
    Compute the loss for masked node and adjacency matrix prediction.

    :param node_logits: Logits for node predictions. Shape: [batch_size, sequence_length, nodes_vocab_size]
    :param true_nodes: Ground truth for nodes. Shape: [batch_size, sequence_length]
    :param node_masks: Binary masks indicating where nodes were masked. Shape: [batch_size, sequence_length]
    
    :param adj_logits: Logits for adjacency matrix predictions. Shape: [batch_size, sequence_length, sequence_length, edge_vocab_size]
    :param true_adj: Ground truth for adjacency matrix. Shape: [batch_size, sequence_length, sequence_length]
    :param adj_masks: Binary masks indicating where adjacency matrix entries were masked. Shape: [batch_size, sequence_length, sequence_length]

    :return: Total loss (combined node and adjacency matrix prediction loss)
    """
    # Node prediction loss
    node_logits_masked = node_logits[node_masks]
    true_nodes_masked = true_nodes[node_masks]
    node_loss = F.cross_entropy(node_logits_masked, true_nodes_masked, reduction='mean')

    # Adjacency matrix prediction loss
    adj_logits_masked = adj_logits[adj_masks]
    true_adj_masked = true_adj[adj_masks]
    adj_loss = F.cross_entropy(adj_logits_masked, true_adj_masked, reduction='mean')

    # Combine the losses
    total_loss = node_loss + adj_loss

    return total_loss

