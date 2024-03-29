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

        encoder_list = [Encoder(k1, n1, enc_layers, enc_hidden_size).to(device), Encoder(k2, n2, enc_layers, enc_hidden_size).to(device)]
        decoder_list = [Decoder(k2, n2, dec_layers, dec_hidden_size).to(device), Decoder(k1, n1, dec_layers, dec_hidden_size).to(device)]
        enc_optimizer_list = [torch.optim.Adam(enc.parameters(), lr=encoder_learning_rate) for enc in encoder_list]
        dec_optimizer_list = [torch.optim.Adam(dec.parameters(), lr=decoder_learning_rate) for dec in decoder_list]
        loss_fn = nn.BCEWithLogitsLoss()
        dataset = InfWordDataset(k1, k2, num_samples, device)
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
                    loss = pipeline(encoder_list, decoder_list, enc_optimizer_list, dec_optimizer_list, loss_fn, iws, freeze_enc=True, freeze_dec=False)

            pbar.set_description(f"Decoder training, loss: {loss.item():.6f}, epoch: {global_ep_idx}")
            print()
            
            for epoch in range(num_encoder_epochs):
                dataset.update_dataset()
                for iws in dataloader:
                    loss = pipeline(encoder_list, decoder_list, enc_optimizer_list, dec_optimizer_list, loss_fn, iws, freeze_enc=False, freeze_dec=True)

            pbar.set_description(f"Encoder training, loss: {loss.item():.6f}, epoch: {global_ep_idx}")
            print()
            # print(f'{time.time()-start_time}s per global epoch')

        torch.save(
            {
                'encoder0': encoder_list[0].state_dict(),
                'encoder1': encoder_list[1].state_dict(),
                'decoder0': decoder_list[0].state_dict(),
                'decoder1': decoder_list[1].state_dict()
            },
            f'model_product_{snr_db.cpu().numpy():.2f}db_clip.pth')
    elif sys.argv[1] == 'test':
        snr_vs_ber()
