'''
Plotting of data graphs
'''

from cProfile import label
import matplotlib.pyplot as plt
from model import *


PATH = "/Users/gennady/Skoltech/Research/networks/"

def snr_vs_ber(snr = np.linspace(1, 5, 9),
               model_list = np.linspace(1.0, 1.0, 1),
               clip_list = ["no_clip", "clip"],
               ):
    '''
    Generate SNR vs FER plots for different models
    '''


    ber_dict = dict()

    for model in model_list:
        for clip in clip_list:
            encoder = Encoder(k, n, enc_layers, enc_hidden_size).to(device)
            decoder = Decoder(k, n, dec_layers, dec_hidden_size).to(device)
            encoder.load_state_dict(torch.load(PATH + f'model_{model:.2f}db_{clip}.pth')['encoder'])
            decoder.load_state_dict(torch.load(PATH + f'model_{model:.2f}db_{clip}.pth')['decoder'])
            
            b=[]

            for v, s in enumerate(snr):
                dataset = InfWordDataset(k, num_samples, device)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

                dataset.update_dataset()
                sz = 0
                for iws in dataloader:
                    encoded = encoder(iws)
                    enc_norm = normalize_power(encoded)
                    enc_clip = enc_norm
                    enc_norm_noise = add_noise(enc_clip, torch.tensor(s, dtype=torch.float, device=device))
                    decoded = decoder(enc_norm_noise)
                    decoded = binarize(-1*decoded.detach().numpy())
                    sz += np.sum(decoded != iws.detach().numpy())
                b.append(sz / (num_samples*k))
            ber_dict[(str(model), clip)] = b


    b = []
    for v, s in enumerate(snr):
        dataset = InfWordDataset(k, num_samples, device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        dataset.update_dataset()
        sz = 0
        for iws in dataloader:
            bin_enc = to_const_points(iws.detach().clone())
            enc_norm = normalize_power(bin_enc)
            enc_norm_noise = add_noise(enc_norm, torch.tensor(s, dtype=torch.float, device=device))
            decoded = binarize(enc_norm_noise.detach().numpy())
            sz += np.sum(decoded != iws.detach().numpy())    
        b.append(sz / (num_samples*k))
    ber_dict['Uncoded'] = b
    
    b = []
    encoder = Encoder(k, n, enc_layers, enc_hidden_size).to(device)
    encoder.load_state_dict(torch.load(PATH + 'model_1.00db_clip.pth')['encoder'])
    codebook = gen_codebook(encoder)
    for v, s in enumerate(snr):
        dataset = InfWordDataset(k, num_samples, device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        dataset.update_dataset()
        sz = 0
        for iws in dataloader:
            bin_enc = encoder(iws.detach().clone())
            enc_norm = normalize_power(bin_enc)
            enc_norm_noise = add_noise(enc_norm, torch.tensor(s, dtype=torch.float, device=device))
            decoded = nearest_point(enc_norm_noise.detach().numpy(), codebook)
            sz += np.sum(decoded != iws.detach().numpy())
        b.append(sz / (num_samples*k))
    ber_dict['Minimal distance decoding'] = b
    
    ber_dict['Gaussian (16, 8) code'] = [4.08333333e-03, 1.75000000e-03, 5.00000000e-04, 3.12500000e-04, 6.25000000e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
    
    plt.figure(figsize=(10, 5))
    for q in ber_dict:
        if len(q) != 2:
            plt.plot(snr, ber_dict[q], label = q, linestyle='--', marker='o')
        else:
            plt.plot(snr, ber_dict[q], label = f"SNR: {q[0]}, {q[1]}", linestyle='--', marker='d')
    plt.grid()
    plt.xlabel("SNR")
    plt.ylabel("BER")
    plt.title("BER vs SNR for different models")
    plt.yscale('log')
    plt.yticks([1e-3, 1e-2, 1e-1, 1])
    plt.legend()
    plt.show()
        
def encoded_distribution():
    '''
    Generate the distribution density of the encoded data
    '''
    
    
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

    plt.show()
    
def pdf_distribution():
    '''
    Generate the PDF of the eccoded data distance
    '''
    
    
    encoder = Encoder(k, n, enc_layers, enc_hidden_size).to(device)
    plt.figure(figsize=(10, 5))
    
    for clip in ['clip', 'no_clip']:
        encoder.load_state_dict(torch.load(PATH + f'model_1.00db_{clip}.pth')['encoder'])
        _, codebook = gen_codebook(encoder)
        dist = np.zeros((codebook.shape[0] * (codebook.shape[0] - 1) ), dtype=np.float32)
        q = 0
        for i in range(codebook.shape[0]):
            for j in range(codebook.shape[0]):
                if i != j:
                    dist[i*codebook.shape[0] + j - q] = np.linalg.norm(codebook[i] - codebook[j])
                else:
                    q += 1

        #plt.hist(dist, bins=len(dist)//10, density=True, cumulative=True, )
        dist = np.sort(dist)
        cum_sum_dist = np.cumsum(dist)
        plt.plot(dist, cum_sum_dist / cum_sum_dist[-1], label = clip)
    #plt.plot(np.arange(0, len(dist), 1), np.cumsum(np.sort(dist)) / len(dist))
    plt.grid() 
    plt.legend()
    plt.xlabel("Distance")
    plt.ylabel("CDF")
    plt.title("CDF of the encoded data distance")
    plt.show()
