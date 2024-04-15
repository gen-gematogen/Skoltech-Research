'''
Implementation of the Network architecture
'''

import itertools
import numpy as np
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')


class Encoder(nn.Module):
    def __init__(self, k, n, enc_layers, hidden_size):
        super().__init__()

        layers = []

        layers.append(nn.Linear(k, hidden_size))
        layers.append(nn.SELU())
        for _ in range(enc_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_size, n))

        self.fcnn = nn.Sequential(
            *layers
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.fcnn(x)
        return logits

class Decoder(nn.Module):
    def __init__(self, k, n, dec_layers, hidden_size):
        super().__init__()

        layers = []

        # # first decoder 
        layers.append(nn.Linear(k, hidden_size))
        layers.append(nn.SELU())
        # last decoder
        for _ in range(dec_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_size, n))
        
        #layers.append(nn.Tanh())

        self.fcnn = nn.Sequential(
            *layers
        )
        
    def forward(self, x):
        # x = self.flatten(x)
        logits = self.fcnn(x)
        return logits
    
class InfWordDataset(torch.utils.data.Dataset):
    def __init__(self, k1, k2, num_samples, device):
        self.k1 = k1
        self.k2 = k2
        self.num_samples = num_samples
        self.device = device

    def update_dataset(self):
        self.iws = torch.randint(low=0, high=2, size=(self.num_samples, self.k1*self.k2), dtype=torch.float, device=self.device)
        self.iws = self.iws.reshape((self.num_samples, self.k1, self.k2))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.iws[idx]

def encoder_pipeline(encoder_pair, x):
    for enc in encoder_pair:
        x = torch.permute(x, (0,2,1))
        x = enc(x)
    encoded = x.reshape((batch_size, n1*n2))
    enc_norm = normalize_power(encoded)
    return enc_norm

def decoder_pipeline(decoder_list, y):
    for i, decoder_pair in enumerate(decoder_list[:-1]):
        if i == 0:
            y_2 = decoder_pair[0](y).reshape(-1, F * n1, n2)
        else:
            y_2_out = decoder_pair[0](y_22_in)
            y_2 = (y_2_out - y_21_in).reshape(-1, F * n1, n2)
            
        y_1_in = torch.cat((y, y_2), 1).permute((0, 2, 1))
        y_1 = decoder_pair[1](y_1_in).permute((0,2,1))
        y_21_in = (y_1 - y_2).reshape(-1, n1, F * n2)
        y_22_in = torch.cat((y, y_21_in), 2)
    y_2 = decoder_list[-1][0](y_22_in).reshape(-1, F * n1, k2)
    y_1 = decoder_list[-1][1](y_2.permute((0, 2, 1)))
    u = y_1.reshape(-1, k1, k2)
    return u

def add_noise(message, snr_db):
    sigma = torch.sqrt(1/(2*10**(snr_db/10)))
    noise = torch.randn(message.shape, device=message.device) * sigma
    return message + noise
    

def normalize_power(tensor):
    #print(tensor.shape)
    norm = torch.norm(tensor, p=2, dim=1, keepdim=True)
    normalized_vector = tensor*torch.sqrt(torch.tensor(tensor.shape[1], device=tensor.device)) / norm
    return normalized_vector

def binarize(tensor):
    tensor[tensor >= 0] = 1.0
    tensor[tensor < 0] = 0.0
    return tensor

def to_const_points(tensor):
    tensor[tensor == 0] = -1
    return tensor

def nearest_point(tensor, codebook):
    words, enc_words = codebook
    d = np.zeros((tensor.shape[0], words.shape[1]))
    for i in range(tensor.shape[0]):
        pos = np.argmin(np.linalg.norm(enc_words - tensor[i], axis=1))
        d[i] = words[pos]
    return d

def gen_codebook(encoder_list):
    # codebook: encoded word -> uncoded word
    words = np.array(list(itertools.product([0, 1], repeat=k1*k2)), dtype=np.float32).reshape(-1, k1, k2)
    cur = torch.tensor(words, dtype=torch.float, device=device)
    for enc in encoder_list:
        cur = enc(cur)
        cur = torch.permute(cur, (0,2,1))    
    
    return (words, cur.detach().numpy())
