import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

import torchaudio
import torchaudio.functional as FA
import torchaudio.transforms as T


# parameters
n_mels = 80
n_embedding = 512
n_blocks = 5
n_t = 10 # TO DEFINE
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # constant accessible (not trainable)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C ** -0.5 # divide by the square root of the size of the embedding
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=1)

        wei = self.dropout(wei)

        v = self.value(x)
        h = wei @ v
        return h
    

class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embedding, n_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.cat([h(x) for h in self.heads], dim=-1)
        h = self.proj(h)
        h = self.dropout(h)
        return h


class FeedForward(nn.Module):
    ''' Feed Forward Module '''
    def __init__(self, input_size, output_size, hidden_sizes):
        super(FeedForward, self).__init__()
        layers = []
        input_dim = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.GELU())
            input_dim = hidden_size
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = MultiHead(n_heads, head_size)
        self.ff = FeedForward(n_embedding, n_embedding, hidden_sizes)
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        h = self.mha(x)
        h = self.ln1(x + h)
        h = self.ff(h)
        h = self.ln2(h + h)
        return h

def create_sin_embdng(n_t, n_embedding):
    ''' Create Sinusoidal Positional Embedding '''
    pos_embedding = torch.zeros(n_t, n_embedding)
    for pos in range(n_t):
        for i in range(0, n_embedding, 2):
            pos_embedding[pos, i] = np.sin(pos / 10000 ** (2 * i / n_embedding))
            pos_embedding[pos, i + 1] = np.cos(pos / 10000 ** (2 * (i + 1) / n_embedding))
    return pos_embedding


class SpeechEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layer
        # n_mels, T -> n_mels, 0 ; n_mels, 1 ; ... n_mels, T
        
        self.conv1 = nn.Conv1d(in_channels=n_mels, out_channels=n_embedding, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=n_embedding, out_channels=n_embedding, stride=2, kernel_size=3, padding=1)
        
        # n_embedding, T -> n_embedding, 0; n_embedding, 1; ... n_embedding, T
        
        # Add Sinuosidal positional embedding
        self.pos_embedding = create_sin_embdng(n_t, n_embedding)

        blocks = []
        for _ in range(n_blocks):
            blocks.append(AttentionBlock())
        self.blocks = nn.Sequential(*blocks)

        self.attention_ln = nn.LayerNorm(n_embedding)

        self.mlp = FeedForward(n_embedding, n_embedding, [2048])
        
        self.mlp_ln = nn.LayerNorm(n_embedding)
        
    def forward(self):
        pass

class DeepSpeechModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpeechEncoder()
        
        self.decoder # ...