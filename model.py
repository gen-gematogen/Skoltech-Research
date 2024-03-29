'''
Implementation of the Network architecture
'''

import itertools
import numpy as np
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k1 = 8
n1 = 16
k2 = 8
n2 = 16
enc_layers = 3
dec_layers = 3
enc_hidden_size = 96
dec_hidden_size = 96
num_decoder_epochs = 400
num_encoder_epochs = 200
total_num_epochs = 10#30
min_clip = -1.6
max_clip = 1.6
num_samples = int(5e2)#int(5e3)
batch_size = int(5e1)#int(5e2)
snr_db = torch.tensor(2, dtype=torch.float, device=device)
encoder_learning_rate = 1e-4#2e-5
decoder_learning_rate = 1e-4#2e-5


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
    
def pipeline(encoder_list, decoder_list, enc_optimizer_list, dec_optimizer_list, loss_fn, iws, freeze_enc = True, freeze_dec = True):
    if not freeze_dec:
        for dec in dec_optimizer_list:
            dec.zero_grad()
    if not freeze_enc:
        for enc in enc_optimizer_list:
            enc.zero_grad()

    cur = iws.detach().clone() 

    for enc in encoder_list:
        cur = enc(cur)
        cur = torch.permute(cur, (0,2,1))
    encoded = cur
    encoded = encoded.reshape((batch_size, n1*n2))
    
    enc_norm = normalize_power(encoded)
    enc_clip = torch.clamp(enc_norm, min_clip, max_clip)
    enc_norm_noise = add_noise(enc_clip, snr_db)
    
    cur = enc_norm_noise
    cur = cur.reshape((batch_size, n1, n2))

    for dec in decoder_list:
        cur = torch.permute(cur, (0,2,1))
        cur = dec(cur)
    decoded = cur
    
    loss = loss_fn(-1*decoded, iws)
    loss.backward()
    
    if not freeze_dec:
        for dec_opt in dec_optimizer_list:
            dec_opt.step()
    if not freeze_enc:
        for enc_opt in enc_optimizer_list:
            enc_opt.step()
    
    return loss


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