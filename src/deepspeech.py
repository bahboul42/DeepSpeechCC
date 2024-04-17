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
dropout = .3
n_heads = 8


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
    
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(n_t, n_t))) # constant accessible (not trainable)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, n_t, n_embedding = x.shape

        # (n_embedding, head_size) * B, ith, n_embedding -> B, ith, head_size
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * n_embedding ** -0.5 # divide by the square root of the size of the embedding
        wei = wei.masked_fill(self.tril[:n_t, :n_t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=1)

        wei = self.dropout(wei)

        v = self.value(x)
        h = wei @ v # B, n_t, n_t @ B, n_t, head_size := B, n_t, head_size
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

class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = MultiHead(n_heads, head_size)
        self.ff = FeedForward(n_embedding, n_embedding, [2048])
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        h = self.mha(x)
        h = self.ln1(x + h)
        h = self.ff(h)
        h = self.ln2(h + h)
        return h

def create_sin_embdng(n_t, n_embedding): # REFINE AND TENSORIZE
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
        self.gelu = nn.GELU()
        
        # n_embedding, T -> n_embedding, 0; n_embedding, 1; ... n_embedding, T
        # Sinusoidal Positional Embedding
        pos_embedding = create_sin_embdng(n_t, n_embedding)
        self.register_buffer('pos_embedding', pos_embedding)

        # Attention Blocks
        blocks = []
        for _ in range(n_blocks):
            blocks.append(AttentionBlock())
        self.blocks = nn.Sequential(*blocks)

        # Last Layer Normalization
        self.ln = nn.LayerNorm(n_embedding)
    
    def forward(self, x):
        # x: (B, n_mels, T)
        x = self.gelu(self.conv1(x)) # (B, n_embedding, T)
        x = self.gelu(self.conv2(x)) # (B, n_embedding, T)
        
        # x: (B, n_embedding, T) , pos_embedding : (T, n_embedding)
        x = x.permute(0, 2, 1) # (B, T, n_embedding)
        x = x + self.pos_embedding

        for block in self.blocks:
            x = block(x)
        
        return self.ln(x)

class DeepSpeechModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpeechEncoder()
        
        self.decoder # ...