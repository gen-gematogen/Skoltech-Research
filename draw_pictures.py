'''
Plotting of data graphs
'''

from cProfile import label
import matplotlib.pyplot as plt
from model import *


PATH = "/home/shutkov/Skoltech-Research/networks/"

def snr_vs_ber(snr = np.linspace(1, 5, 9),
               model_list = np.linspace(1.0, 1.0, 1),
               clip_list = ["no_clip", "clip"],
               ):
    '''
    Generate SNR vs FER plots for different models
    '''

    clip_list = ['5']
    model_list = [2.0]
    snr = np.linspace(0, 5, 10)
    ber_dict = dict()
    torch.no_grad()

    for model in model_list:
        for clip in clip_list:
            encoder_list = [Encoder(k1, n1, enc_layers, enc_hidden_size).to(device), Encoder(k2, n2, enc_layers, enc_hidden_size).to(device)]
            
            decoder_list = []
            for i in range(I):
                if i == 0:
                    decoder_list.append([Decoder(n2, F * n2, dec_layers, dec_hidden_size).to(device), Decoder((1 + F) * n1, F * n1, dec_layers, dec_hidden_size).to(device)])
                elif i == I - 1:
                    decoder_list.append([Decoder((1 + F) * n2, F * k2, dec_layers, dec_hidden_size).to(device), Decoder(F * n1, k1, dec_layers, dec_hidden_size).to(device)])
                else:
                    decoder_list.append([Decoder((1 + F) * n2, F * n2, dec_layers, dec_hidden_size).to(device), Decoder((1 + F) * n1, F * n1, dec_layers, dec_hidden_size).to(device)])
            
            for i in range(len(encoder_list)):
                encoder_list[i].load_state_dict(torch.load(PATH + f'model_product_article_{model:.2f}db_{clip}.pth')[f'encoder{i}'])
                encoder_list[i].eval()
            for i in range(len(decoder_list)):
                decoder_list[i][0].load_state_dict(torch.load(PATH + f'model_product_article_{model:.2f}db_{clip}.pth')[f'decoder{i}_0'])
                decoder_list[i][1].load_state_dict(torch.load(PATH + f'model_product_article_{model:.2f}db_{clip}.pth')[f'decoder{i}_1'])
                decoder_list[i][0].eval()
                decoder_list[i][1].eval()
            b=[]

            for v, s in enumerate(snr):
                snr_db = torch.tensor(s, dtype=torch.float, device=device)
                dataset = InfWordDataset(k1, k2, num_samples, device)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

                dataset.update_dataset()
                sz = 0
                for iws in dataloader:
                    enc_data = encoder_pipeline(encoder_list, iws.detach().clone())
                    # enc_data = torch.clamp(enc_data, min_clip, max_clip)
                    enc_data_noise = add_noise(enc_data, snr_db).reshape(-1, n1, n2)
                    dec_data = decoder_pipeline(decoder_list, enc_data_noise)
                    
                    decoded = binarize(-1*dec_data.detach())
                    sz += torch.sum(decoded != iws)
                b.append((sz / (num_samples*k1*k2)).item())
            ber_dict[(str(model), clip)] = b


    b = []
    for v, s in enumerate(snr):
        dataset = InfWordDataset(k1, k2, num_samples, device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        dataset.update_dataset()
        sz = 0
        for iws in dataloader:
            bin_enc = to_const_points(iws.detach().clone())
            bin_enc = bin_enc.reshape((batch_size, k1*k2))
            enc_norm = normalize_power(bin_enc)
            enc_norm_noise = add_noise(enc_norm, torch.tensor(s, dtype=torch.float, device=device))
            decoded = binarize(enc_norm_noise.detach())
            decoded = decoded.reshape((batch_size, k1, k2))
            sz += torch.sum(decoded != iws)    
        b.append((sz / (num_samples*k1*k2)).item())
    ber_dict['Uncoded'] = b
    
    # b = []
    # encoder_list = [Encoder(k1, n1, enc_layers, enc_hidden_size).to(device), Encoder(k2, n2, enc_layers, enc_hidden_size).to(device)]
    # for i in range(len(encoder_list)):
    #     encoder_list[i].load_state_dict(torch.load(PATH + f'model_product_{model:.2f}db_{clip}.pth')[f'encoder{i}'])
    # codebook = gen_codebook(encoder_list)
    # for v, s in enumerate(snr):
    #     dataset = InfWordDataset(k1, k2, num_samples, device)
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    #     dataset.update_dataset()
    #     sz = 0
    #     for iws in dataloader:
    #         cur = iws.detach().clone() 
    #         for enc in encoder_list:
    #             cur = enc(cur)
    #             cur = torch.permute(cur, (0,2,1))
    #         bin_enc = cur.reshape((batch_size, n1*n2))
    #         enc_norm = normalize_power(bin_enc)
    #         enc_norm_noise = add_noise(enc_norm, torch.tensor(s, dtype=torch.float, device=device))
    #         decoded = nearest_point(enc_norm_noise.detach().numpy(), codebook).reshape((batch_size, k1, k2))
    #         sz += np.sum(decoded != iws.detach().numpy())
    #     b.append(sz / (num_samples*k1*k2))
    # ber_dict['Minimal distance decoding'] = b
    
    # ber_dict['Gaussian (16, 8) code'] = [4.08333333e-03, 1.75000000e-03, 5.00000000e-04, 3.12500000e-04, 6.25000000e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
    
    plt.figure(figsize=(10, 5))
    for q in ber_dict:
        if len(q) != 2:
            plt.plot(snr, ber_dict[q], label = q, linestyle='--', marker='o')
        else:
            plt.plot(snr, ber_dict[q], label = f"SNR: {q[0]}, {q[1]}", linestyle='--', marker='d')
    plt.grid()
    plt.xlabel("SNR")
    plt.ylabel("BER")
    plt.title("BER vs SNR for topology from article")
    plt.yscale('log')
    #plt.yticks([1e-3, 1e-2, 1e-1, 1])
    plt.legend()
    plt.savefig("ber_vs_snr.png")
    #plt.show()
        
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
