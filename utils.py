import torch
from torch import nn
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import sys
import itertools
import numpy as np

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
    d = np.zeros((tensor.shape[0]))
    for i in range(tensor.shape[0]):
        d[i] = codebook[np.argmin(np.linalg.norm(tensor[i] - codebook, axis=1))]
    return d
    

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        snr_db = torch.tensor(float(sys.argv[2]), dtype=torch.float, device=device)

        encoder = Encoder(k, n, enc_layers, enc_hidden_size).to(device)
        decoder = Decoder(k, n, dec_layers, dec_hidden_size).to(device)
        enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
        dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        dataset = InfWordDataset(k, num_samples, device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        pbar = tqdm(range(total_num_epochs))

        for global_ep_idx in pbar:
            start_time = time.time()
            if global_ep_idx % 5 == 0:
                min_clip += 0.1
                max_clip -= 0.1
            for epoch in range(num_decoder_epochs):
                dataset.update_dataset()
                for iws in dataloader:
                    dec_optimizer.zero_grad()
                    encoded = encoder(iws)
                    enc_norm = normalize_power(encoded)
                    
                    enc_clip = torch.clamp(enc_norm, min_clip, max_clip)
                    
                    enc_norm_noise = add_noise(enc_clip, snr_db)
                    decoded = decoder(enc_norm_noise)
                    loss = loss_fn(-1*decoded, iws)
                    loss.backward()
                    dec_optimizer.step()

            pbar.set_description(f"Decoder training, loss: {loss.item():.6f}, epoch: {global_ep_idx}")
            print()
            
            for epoch in range(num_encoder_epochs):
                dataset.update_dataset()
                for iws in dataloader:
                    enc_optimizer.zero_grad()
                    encoded = encoder(iws)
                    enc_norm = normalize_power(encoded)
                    
                    enc_clip = torch.clamp(enc_clip, min_clip, max_clip)
                    
                    enc_norm_noisy = add_noise(enc_norm, snr_db)
                    decoded = decoder(enc_norm_noisy)
                    loss = loss_fn(-1*decoded, iws)
                    loss.backward()
                    enc_optimizer.step()

            pbar.set_description(f"Encoder training, loss: {loss.item():.6f}, epoch: {global_ep_idx}")
            print()
            # print(f'{time.time()-start_time}s per global epoch')

        torch.save(
            {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict()
            },
            f'model_{snr_db.cpu().numpy():.2f}db_clip.pth')
    elif sys.argv[1] == 'test':
        # --------------------------------------------------------------------------------
        # Generate SNR vs FER plots for different models
        # --------------------------------------------------------------------------------
        
        # snr = np.linspace(1, 5, 9)
        # ber_dict = dict()
    
        # for model in np.linspace(1, 5, 3):
        #     for clip in ["no_clip", "clip"]:
        #         encoder = Encoder(k, n, enc_layers, enc_hidden_size).to(device)
        #         decoder = Decoder(k, n, dec_layers, dec_hidden_size).to(device)
        #         encoder.load_state_dict(torch.load(f'/Users/gennady/Skoltech/Research/networks/model_{model:.2f}db_{clip}.pth')['encoder'])
        #         decoder.load_state_dict(torch.load(f'/Users/gennady/Skoltech/Research/networks/model_{model:.2f}db_{clip}.pth')['decoder'])
                
        #         b=[]
        #         for v, s in enumerate(snr):
        #             dataset = InfWordDataset(k, num_samples, device)
        #             dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        #             # evaluate the encoded data distribution density
        #             dataset.update_dataset()
        #             dencity = dict()
        #             sz = 0
        #             for iws in dataloader:
        #                 encoded = encoder(iws)
        #                 enc_norm = normalize_power(encoded)
        #                 enc_clip = enc_norm#torch.clamp(enc_norm, min_clip, max_clip)
        #                 enc_norm_noise = add_noise(enc_clip, torch.tensor(s, dtype=torch.float, device=device))
        #                 decoded = decoder(enc_norm_noise)
        #                 decoded = binarize(-1*decoded.detach().numpy())
                        
        #                 sz += np.sum(np.any(decoded != iws.detach().numpy(), axis=1))
                    
        #             b.append(sz / (num_samples))
        #         ber_dict[(str(model), clip)] = b

        # b = []
        # for v, s in enumerate(snr):
        #     dataset = InfWordDataset(k, num_samples, device)
        #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        #     dataset.update_dataset()
        #     dencity = dict()
        #     sz = 0
        #     for iws in dataloader:
        #         bin_enc = to_const_points(iws.detach().clone())
        #         enc_norm = normalize_power(bin_enc)
        #         enc_norm_noise = add_noise(enc_norm, torch.tensor(s, dtype=torch.float, device=device))
        #         decoded = binarize(enc_norm_noise.detach().numpy())
        #         sz += np.sum(np.any(decoded != iws.detach().numpy(), axis=1))
                    
        #     b.append(sz / (num_samples))
        # ber_dict['Uncoded'] = b
        
        # for k in ber_dict:
        #     if k == 'Uncoded' or k == 'MSE':
        #         plt.plot(snr, ber_dict[k], label = k, linestyle='--')
        #     else:
        #         plt.plot(snr, ber_dict[k], label = f"Training SNR: {k[0]}, {k[1]}", linestyle='--')
        # plt.grid()
        # plt.xlabel("SNR")
        # plt.ylabel("FER")
        # plt.title("FER vs SNR for different models")
        # plt.yscale('log')
        # plt.yticks([1e-3, 1e-2, 1e-1, 1])
        # plt.legend()
        # plt.show()
        
        # --------------------------------------------------------------------------------
        # Generate the distribution density of the encoded data
        # --------------------------------------------------------------------------------
        
        ax, fig = plt.subplots(1, 2, figsize=(10, 5))
        
        model = 1.00
        for j, clip in enumerate(["no_clip", "clip"]):
            encoder = Encoder(k, n, enc_layers, enc_hidden_size).to(device)
            encoder.load_state_dict(torch.load(f'/Users/gennady/Skoltech/Research/networks/model_{model:.2f}db_{clip}.pth')['encoder'])
            
            dataset = InfWordDataset(k, num_samples, device)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            dataset.update_dataset()
            
            dencity = np.zeros((n))
            
            p=0
            for iws in dataloader:
                p+=1
                encoded = encoder(iws)
                enc_norm = normalize_power(encoded)
                
                dencity += np.mean(enc_norm.detach().numpy(), axis=0)
                        
            dencity /= p
                
            fig[j].bar(range(1, 17), dencity, width=0.5)
            fig[j].set_title(f"Training SNR: {model}db, {clip}")
            fig[j].grid()
            fig[j].set_xlabel("Bin number")
            fig[j].set_ylabel("Mean value")
            fig[j].set_xticks(range(1, 17, 2))
        #plt.grid()
        #plt.xlabel("Vlue")
        #plt.ylabel("Density")
        plt.show()
        

