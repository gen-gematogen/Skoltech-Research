'''
Implementation of the Network architecture
'''

import itertools
import numpy as np
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 8
n = 16
enc_layers = 3
dec_layers = 3
enc_hidden_size = 96
dec_hidden_size = 96
num_decoder_epochs = 400
num_encoder_epochs = 200
total_num_epochs = 30
min_clip = -1.6
max_clip = 1.6
num_samples = int(5e3)
batch_size = int(5e2)
snr_db = torch.tensor(2, dtype=torch.float, device=device)
encoder_learning_rate = 2e-5
decoder_learning_rate = 2e-5


class Encoder(nn.Module):
    def __init__(self, k, n, enc_layers, hidden_size):
        super().__init__()
        # self.flatten = nn.Flatten()

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
        layers.append(nn.Linear(n, hidden_size))
        layers.append(nn.SELU())
        # last decoder
        for _ in range(dec_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_size, k))
        
        #layers.append(nn.Tanh())

        self.fcnn = nn.Sequential(
            *layers
        )
        
    def forward(self, x):
        # x = self.flatten(x)
        logits = self.fcnn(x)
        return logits
    
class InfWordDataset(torch.utils.data.Dataset):
    def __init__(self, k, num_samples, device):
        self.k = k
        self.num_samples = num_samples
        self.device = device

    def update_dataset(self):
        self.iws = torch.randint(low=0, high=2, size=(self.num_samples, self.k), dtype=torch.float, device=self.device)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.iws[idx]


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

def gen_codebook(encoder):
    # codebook: encoded word -> uncoded word
    words = np.array(list(itertools.product([0, 1], repeat=k)), dtype=np.float32)
    enc = encoder(torch.tensor(words, dtype=torch.float, device=device))
        
    return (words, enc.detach().numpy())