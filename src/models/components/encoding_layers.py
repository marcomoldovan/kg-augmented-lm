import torch
import torch.nn as nn



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: Tensor of shape [batch_size, seq_length, d_model]
        :return: Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return x
    
    
class LookupTableModalityEmbedding(nn.Module):
    def __init__(self, feature_dim, num_modalities):
        super(LookupTableModalityEmbedding, self).__init__()

        # Embedding layer for modality embeddings
        self.modality_embeddings = nn.Embedding(num_modalities, feature_dim)

    def forward(self, features, modality_ids):
        """
        :param features: [batch_size, sequence_length, feature_dim]
        :param modality_ids: [batch_size, sequence_length] with values in [0, num_modalities-1]
        :return: Features with modality embeddings added.
        """
        embeddings = self.modality_embeddings(modality_ids)
        return features + embeddings


class SinusoidalModalityEmbedding(nn.Module):
    def __init__(self, feature_dim, num_modalities):
        super(SinusoidalModalityEmbedding, self).__init__()

        # Create sinusoidal modality embeddings
        position = torch.arange(0, num_modalities, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / feature_dim))
        sinusoidal_embedding = torch.zeros(num_modalities, feature_dim)
        sinusoidal_embedding[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_embedding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('sinusoidal_embedding', sinusoidal_embedding)

    def forward(self, features, modality_ids):
        """
        :param features: [batch_size, sequence_length, feature_dim]
        :param modality_ids: [batch_size, sequence_length] with values in [0, num_modalities-1]
        :return: Features with sinusoidal modality embeddings added.
        """
        embeddings = self.sinusoidal_embedding[modality_ids]
        return features + embeddings