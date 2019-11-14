"""
Convert the trained entity embedding into item feature pickle files
as the input data for downstream recsys models
The saved pickle file contains a torch.Tensor object with shape=(#items, #embed_dim)
"""
import argparse
import numpy as np
import os
import pickle
import torch as th

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert entity embeddings to item features as a pickle file")
    parser.add_argument('--data_name', nargs='?', default='',
                        help='the data dir name')
    parser.add_argument('--kge_model', nargs='?', default='',
                        help='the data dir name')
    parser.add_argument('--kg_file', type=str, default='kg.txt',
                        help='the input kg txt file name')
    parser.add_argument('--n_item', type=int, help='(Required) the total number of items')
    args = parser.parse_args()
    assert args.n_item is not None

    all_entity_emb = np.load(os.path.join("KG_trained_embed",
                                          '{}_{}_entity.npy'.format(args.data_name, args.kge_model)))
    item_entity_emb_fea_np = all_entity_emb[:args.n_item, :]
    print(item_entity_emb_fea_np)
    pickle.dump(th.Tensor(item_entity_emb_fea_np),
                open('{}_entity_embed_features.pkl'.format(args.data_name), 'wb'))

    ### testing
    # f = open('{}_entity_embed_features.pkl'.format(args.data_name), "rb")
    # data = pickle.load(f)
    # f.close()
    # print(data)
