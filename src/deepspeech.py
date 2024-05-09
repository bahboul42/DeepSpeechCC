import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import pandas as pd
from tqdm import tqdm

import torchaudio
import torchaudio.functional as FA
import torchaudio.transforms as T

from transformers import GPT2LMHeadModel, GPT2Tokenizer


# parameters
n_mels = 80
n_embedding = 512
n_blocks = 5
n_t = int(626/2) # TO DEFINE
dropout = 0.2 #.3
n_heads = 8
lora_dim = 32


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
        head_size = n_embedding // n_heads
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
        # n_mels, 2*T -> n_mels, 0 ; n_mels, 1 ; ... n_mels, 2*T
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


class GPT2Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = GPT2LMHeadModel.from_pretrained('gpt2')
        n_llm_embd = self.llm.config.n_embd
        self.fc = nn.Linear(n_embedding, n_llm_embd)

    def forward(self, x):
        #x = self.fc(x)
        return self.llm(inputs_embeds=x)['logits']


class DeepSpeechCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpeechEncoder()
        self.decoder = GPT2Decoder()

        self.embder = nn.Linear(n_embedding, self.decoder.llm.config.n_embd)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #self.tokenizer.add_special_tokens({'pad_token': '<|pad|>', 'bos_token': '<|sot|>', 'eos_token': '<|eot|>'})        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.wte = self.decoder.llm.transformer.wte
        self.vocab_size = self.decoder.llm.config.vocab_size

        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward(self, x, y=None):
        B, n_mels, T = x.shape
        x = self.encoder(x) # encode mel spectrogram
        x = self.embder(x) # embed encoded mel spectrogram

        if y is not None:
            y_encoded = self.wte(y) # encode text
            x = torch.cat([x, y_encoded], dim=1)
        
        return self.decoder(x)


    def label_parser(self, y):
        B = len(y)
        #_tokens = self.tokenizer(y)['input_ids']
        #lens = [len(_token) for _token in _tokens]
        #max_len = max(lens)
        tokens = self.tokenizer(y, padding=True, return_tensors='pt')['input_ids']
        # gt_matrix = torch.zeros(B, max_len, self.vocab_size, dtype=torch.int)
        # # Loop over tokens in batch
        # for i, seq in enumerate(tokens):
        #     # Set indices in gt_matrix to 1 at the positions specified by token indices
        #     gt_matrix[i, range(max_len), seq] = 1
        return tokens
    
class CommonVoiceDataset(torch.utils.data.Dataset):#

    def __init__(self, root_dir, mode='train', transform=None, frac=(0.6, 0.3, 0.1), sample=1.0, _tsv_loc=f'validated_cleaned.tsv', _pt_loc='/melspecs/', _random_seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self._tsv_loc = _tsv_loc
        self._pt_loc = _pt_loc
        self._random_seed = _random_seed

        # Load the data
        self.df = pd.read_csv(self._tsv_loc, sep='\t').sample(frac=sample, random_state=self._random_seed)

        self.data = self.df['name'].to_list()
        self.labels = self.df['sentence'].to_list()

        train_frac, val_frac, test_frac = frac

        len_df = len(self.data)

        # Split the data
        np.random.seed(self._random_seed)
        #np.random.shuffle(self.data)
        #np.random.shuffle(self.labels)

        train_size = int(len_df * train_frac)
        val_test_size = len_df - train_size
        val_size = int(len_df * val_frac)
        test_size = val_test_size - val_size

        train_data, val_test_data = torch.utils.data.random_split(self.data, [train_size, val_test_size])
        train_labels, val_test_labels = torch.utils.data.random_split(self.labels, [train_size, val_test_size])
        
        val_data, test_data = torch.utils.data.random_split(val_test_data, [val_size, test_size])
        val_labels, test_labels = torch.utils.data.random_split(val_test_labels, [val_size, test_size])

        if mode == 'train':
            self.data = train_data
            self.labels = train_labels
        elif mode == 'val':
            self.data = val_data
            self.labels = val_labels
        elif mode == 'test':
            self.data = test_data
            self.labels = test_labels
        else:
            raise ValueError('mode must be either \'train\', \'val\', or \'test\'')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.load(self.root_dir + self._pt_loc + self.data[idx]).squeeze(0)
        if self.transform:
            data = self.transform(data)
        return data, self.labels[idx]
