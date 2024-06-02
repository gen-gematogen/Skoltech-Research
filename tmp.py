import json

with open("configs/model_product_article_2.00db_5.json", 'w') as out_f:
    p = dict()
    p['F'] = 3
    p['I'] = 4
    p['k1'] = 5
    p['n1'] = 10
    p['k2'] = 5
    p['n2'] = 10
    p['enc_layers'] = 3
    p['dec_layers'] = 3
    p['enc_hidden_size'] = 100
    p['dec_hidden_size'] = 125
    p['num_decoder_epochs'] = 500
    p['num_encoder_epochs'] = 100#100
    p['total_num_epochs'] = 30
    p['num_samples'] = int(1e5)
    p['batch_size'] = int(5e3)
    p['encoder_learning_rate'] = 2e-4
    p['decoder_learning_rate'] = 2e-4

    json.dump(p, out_f)
