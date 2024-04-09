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
            #if global_ep_idx % 5 == 0:
            #    min_clip += 0.1
            #    max_clip -= 0.1
            for epoch in range(num_decoder_epochs):
                dataset.update_dataset()
                for iws in dataloader:
                    dec_optimizer.zero_grad()
                    encoded = encoder(iws)
                    enc_norm = normalize_power(encoded)
                    
                    #enc_clip = torch.clamp(enc_norm, min_clip, max_clip)
                    
                    enc_norm_noise = add_noise(enc_norm, snr_db)
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
                    
                    #enc_clip = torch.clamp(enc_norm, min_clip, max_clip)
                    
                    enc_norm_noise = add_noise(enc_norm, snr_db)
                    decoded = decoder(enc_norm_noise)
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
        snr_vs_ber()