'''
Main file to process the command line arguments and call required functions
'''

import matplotlib.pyplot as plt
import time
import sys
from tqdm import tqdm
from model import *
from draw_pictures import * 


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        print(f"{device=}")
        snr_db = torch.tensor(float(sys.argv[2]), dtype=torch.float, device=device)

        encoder_list = [Encoder(k1, n1, enc_layers, enc_hidden_size).to(device), Encoder(k2, n2, enc_layers, enc_hidden_size).to(device)]

        decoder_list = []
        for i in range(I):
            if i == 0:
                decoder_list.append([Decoder(n2, F * n2, dec_layers, dec_hidden_size).to(device), Decoder((1 + F) * n1, F * n1, dec_layers, dec_hidden_size).to(device)])
            elif i == I - 1:
                decoder_list.append([Decoder((1 + F) * n2, F * k2, dec_layers, dec_hidden_size).to(device), Decoder(F * n1, k1, dec_layers, dec_hidden_size).to(device)])
            else:
                decoder_list.append([Decoder((1 + F) * n2, F * n2, dec_layers, dec_hidden_size).to(device), Decoder((1 + F) * n1, F * n1, dec_layers, dec_hidden_size).to(device)])

        enc_optimizer_list = [torch.optim.Adam(enc.parameters(), lr=encoder_learning_rate) for enc in encoder_list]

        dec_optimizer_list = []
        for i in range(I):
            dec_optimizer_list.append([torch.optim.Adam(dec.parameters(), lr=decoder_learning_rate) for dec in decoder_list[i]])
            
        loss_fn = nn.BCEWithLogitsLoss()
        dataset = InfWordDataset(k1, k2, num_samples, device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        pbar = tqdm(range(total_num_epochs))

        for global_ep_idx in pbar:
            start_time = time.time()
            #if global_ep_idx % 5 == 0:
            #    min_clip += 0.1
            #    max_clip -= 0.1
            for epoch in tqdm(range(num_decoder_epochs), position=0, leave=True):
                dataset.update_dataset()
                for iws in dataloader:
                    for e in dec_optimizer_list:
                        e[0].zero_grad()
                        e[1].zero_grad()
                    
                    enc_data = encoder_pipeline(encoder_list, iws.detach().clone())
                    #enc_data = torch.clamp(enc_data, min_clip, max_clip)
                    cur_snr = torch.tensor(torch.randn(1) * 3.5 - 2.5 + float(sys.argv[2]), dtype=torch.float, device=device)
                    enc_data_noise = add_noise(enc_data, cur_snr).reshape(-1, n1, n2)
                    dec_data = decoder_pipeline(decoder_list, enc_data_noise)
                    
                    loss = loss_fn(-1*dec_data, iws)
                    loss.backward()
                    
                    for e in dec_optimizer_list:
                        e[0].step()
                        e[1].step()

            pbar.set_description(f"Decoder training, loss: {loss.item():.6f}, epoch: {global_ep_idx}")
            print()
            
            for epoch in range(num_encoder_epochs):
                dataset.update_dataset()
                for iws in dataloader:
                    for e in enc_optimizer_list:
                        e.zero_grad()
                    
                    enc_data = encoder_pipeline(encoder_list, iws.detach().clone())
                    #enc_data = torch.clamp(enc_data, min_clip, max_clip)
                    enc_data_noise = add_noise(enc_data, snr_db).reshape(-1, n1, n2)
                    dec_data = decoder_pipeline(decoder_list, enc_data_noise)
                    
                    loss = loss_fn(-1*dec_data, iws)
                    loss.backward()
                    
                    for e in enc_optimizer_list:
                        e.step()

            pbar.set_description(f"Encoder training, loss: {loss.item():.6f}, epoch: {global_ep_idx}")
            print()
            
        networks = dict()
        for i in range(len(encoder_list)):
            networks[f'encoder{i}'] = encoder_list[i].state_dict()
        for i in range(len(decoder_list)):
            networks[f'decoder{i}_0'] = decoder_list[i][0].state_dict()
            networks[f'decoder{i}_1'] = decoder_list[i][1].state_dict()
            
        torch.save(
            networks,
            f'model_product_article_{snr_db.cpu().numpy():.2f}db_clip.pth')
    elif sys.argv[1] == 'test':
        snr_vs_ber()
