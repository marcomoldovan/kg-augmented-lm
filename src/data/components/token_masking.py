import torch

def token_masking(tensor, pad_id, mask_token=9999, mask_prob=0.15):
    """
    Apply BERT-style masking on the tensor.

    :param tensor: Input tensor to mask.
    :param pad_id: ID used for padding.
    :param mask_token: Token ID to use for the mask.
    :param mask_prob: Probability of masking an entry.
    :param random_replace_prob: Probability of replacing a masked entry with a random token.

    :return: Masked tensor, mask indicating which entries were masked.
    """
    # Determine which entries are eligible for masking (exclude padding entries)
    eligible = (tensor != pad_id) & (tensor != 0) & (torch.rand_like(tensor, dtype=torch.float) < mask_prob)
    
    # Random numbers to determine type of masking (actual mask, random replacement, or unchanged)
    rand_nums = torch.rand_like(tensor, dtype=torch.float)
    
    # Create the masked tensor
    masked_tensor = tensor.clone()
    
    # Replace 80% of eligible with mask_token (this condition satisfies 0.8 probability)
    masked_tensor[(rand_nums < 0.8) & eligible] = mask_token
    
    # Replace 10% of eligible with random token (this condition satisfies 0.1 probability given previous masking)
    max_val = int(tensor.max().item()) + 1  # Assuming token IDs are 0-indexed and contiguous
    random_tokens = torch.randint(1, max_val, tensor.shape, dtype=tensor.dtype, device=tensor.device)
    masked_tensor[(rand_nums >= 0.8) & (rand_nums < 0.9) & eligible] = random_tokens[(rand_nums >= 0.8) & (rand_nums < 0.9) & eligible]
    
    return masked_tensor, eligible

def mask_nodes_and_adj(nodes, adj_matrix, pad_id):
    """
    Apply BERT-style masking on nodes tensor and adjacency matrix.

    :param nodes: Nodes tensor of shape [batch_size, padded_seq_length].
    :param adj_matrix: Adjacency matrix of shape [batch_size, num_nodes, num_nodes].
    :param pad_id: ID used for padding.

    :return: Masked nodes, node masks, masked adjacency matrix, adjacency matrix masks.
    """
    # Masking nodes
    masked_nodes, node_masks = token_masking(nodes, pad_id)
    
    # Masking adjacency matrix
    # Flatten the last two dimensions to treat it as a long sequence for masking
    adj_matrix_flattened = adj_matrix.view(adj_matrix.size(0), -1)
    masked_adj_flattened, adj_masks_flattened = token_masking(adj_matrix_flattened, pad_id)
    
    # Reshape the masked adjacency matrix and its masks back to original shape
    masked_adj = masked_adj_flattened.view(adj_matrix.size())
    adj_masks = adj_masks_flattened.view(adj_matrix.size())

    return masked_nodes, node_masks, masked_adj, adj_masks
