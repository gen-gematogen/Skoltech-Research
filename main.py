'''
Main file to process the command line arguments and call required functions
'''

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import sys
from tqdm import tqdm
from model import *
from draw_pictures import * 


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        print(f"{device=}")
        snr_db = torch.tensor(float(sys.argv[2]), dtype=torch.float, device=device)
        model_name = sys.argv[3]

        print(snr_db)

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
        enc_scheduler_list = [lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-6, verbose=True) for optimizer in enc_optimizer_list]

        dec_optimizer_list = []
        dec_scheduler_list = []
        for i in range(I):
            dec_optimizer_list.append([torch.optim.Adam(dec.parameters(), lr=decoder_learning_rate) for dec in decoder_list[i]])
            dec_scheduler_list.append([lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-6, verbose=True) for optimizer in dec_optimizer_list[-1]])

        loss_fn = nn.BCEWithLogitsLoss()
        dataset = InfWordDataset(k1, k2, num_samples, device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        minimum_loss = float("inf")

        for i in range(len(encoder_list)):
            encoder_list[i].load_state_dict(torch.load(model_name)[f'encoder{i}'])
            encoder_list[i].eval()
        for i in range(len(decoder_list)):
            decoder_list[i][0].load_state_dict(torch.load(model_name)[f'decoder{i}_0'])
            decoder_list[i][1].load_state_dict(torch.load(model_name)[f'decoder{i}_1'])
            decoder_list[i][0].eval()
            decoder_list[i][1].eval()
        

        pbar = tqdm(range(total_num_epochs))

        for global_ep_idx in pbar:
            start_time = time.time()
            for epoch in tqdm(range(num_decoder_epochs), position=0, leave=True):
                dataset.update_dataset()
                dec_loss = 0
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
                    dec_loss += loss.item()

                    for e in dec_optimizer_list:
                        e[0].step()
                        e[1].step()

            pbar.set_description(f"Decoder training, loss: {loss.item():.6f}, epoch: {global_ep_idx}")
            print()
            
            for epoch in tqdm(range(num_encoder_epochs), position=0, leave=True):
                dataset.update_dataset()
                enc_loss = 0
                for iws in dataloader:
                    for e in enc_optimizer_list:
                        e.zero_grad()
                                        
                    enc_data = encoder_pipeline(encoder_list, iws.detach().clone())
                    #enc_data = torch.clamp(enc_data, min_clip, max_clip)
                    enc_data_noise = add_noise(enc_data, snr_db).reshape(-1, n1, n2)
                    dec_data = decoder_pipeline(decoder_list, enc_data_noise)
                    
                    loss = loss_fn(-1*dec_data, iws)
                    loss.backward()
                    enc_loss += loss.item()
                    
                    for e in enc_optimizer_list:
                        e.step()
            
            for s in dec_scheduler_list:
                s[0].step(dec_loss)
                s[1].step(dec_loss)

            for s in enc_scheduler_list:
                s.step(enc_loss)

            pbar.set_description(f"Encoder training, loss: {loss.item():.6f}, epoch: {global_ep_idx}")
            print()

            if enc_loss + dec_loss < minimum_loss:
                dump_model(encoder_list, decoder_list, snr_db)
                minimum_loss = enc_loss + dec_loss

    elif sys.argv[1] == 'test':
        print(f"{device=}")
        snr_vs_ber()

    elif sys.argv[1] == 'tune':
        print("Tuning the model")
        model_name = sys.argv[3]

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
        enc_scheduler_list = [lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-6, verbose=True) for optimizer in enc_optimizer_list]

        dec_optimizer_list = []
        dec_scheduler_list = []
        for i in range(I):
            dec_optimizer_list.append([torch.optim.Adam(dec.parameters(), lr=decoder_learning_rate) for dec in decoder_list[i]])
            dec_scheduler_list.append([lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-6, verbose=True) for optimizer in dec_optimizer_list[-1]])

        loss_fn = nn.BCEWithLogitsLoss()
        dataset = InfWordDataset(k1, k2, num_samples, device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        for i in range(len(encoder_list)):
            encoder_list[i].load_state_dict(torch.load(model_name)[f'encoder{i}'])
            encoder_list[i].eval()
        for i in range(len(decoder_list)):
            decoder_list[i][0].load_state_dict(torch.load(model_name)[f'decoder{i}_0'])
            decoder_list[i][1].load_state_dict(torch.load(model_name)[f'decoder{i}_1'])
            decoder_list[i][0].eval()
            decoder_list[i][1].eval()


        pbar = tqdm(range(total_num_epochs))
        best_loss = float('inf')

        for global_ep_idx in pbar:
            start_time = time.time()
            for epoch in tqdm(range(num_decoder_epochs), position=0, leave=True):
                dataset.update_dataset()
                
                for e in dec_optimizer_list:
                    e[0].zero_grad()
                    e[1].zero_grad()

                for iws in dataloader:
                    enc_data = encoder_pipeline(encoder_list, iws.detach().clone())
                    #enc_data = torch.clamp(enc_data, min_clip, max_clip)
                    cur_snr = torch.tensor(torch.randn(1) * 3.5 - 2.5 + float(sys.argv[2]), dtype=torch.float, device=device)
                    enc_data_noise = add_noise(enc_data, cur_snr).reshape(-1, n1, n2)
                    dec_data = decoder_pipeline(decoder_list, enc_data_noise)

                    loss = loss_fn(-1*dec_data, iws) / (num_samples / batch_size) 
                    loss.backward()

                for e in dec_optimizer_list:
                    e[0].step()
                    e[1].step()

            pbar.set_description(f"Decoder training, loss: {loss.item():.6f}, epoch: {global_ep_idx}")
            print()
            dec_loss = loss.item()

            for epoch in range(num_encoder_epochs):
                    dataset.update_dataset()
                    
                    for e in enc_optimizer_list:
                        e.zero_grad()

                    for iws in dataloader:
                        enc_data = encoder_pipeline(encoder_list, iws.detach().clone())
                        #enc_data = torch.clamp(enc_data, min_clip, max_clip)
                        enc_data_noise = add_noise(enc_data, snr_db).reshape(-1, n1, n2)
                        dec_data = decoder_pipeline(decoder_list, enc_data_noise)

                        loss = loss_fn(-1*dec_data, iws) / (num_samples / batch_size)
                        loss.backward()

                    for e in enc_optimizer_list:
                        e.step()

            for s in dec_scheduler_list:
                    s[0].step()
                    s[1].step()

            for s in enc_scheduler_list:
                    s.step()

            pbar.set_description(f"Encoder training, loss: {loss.item():.6f}, epoch: {global_ep_idx}")
            print()
            enc_loss = loss.item()

            if enc_loss + dec_loss < best_loss:
                best_loss = enc_loss + dec_loss
                dump_model(encoder_list, decoder_list, snr_db)

