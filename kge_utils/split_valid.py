"""
Convert the entire kg data into train, valid, test sets
    python split_valid.py --data_dir sample_FB15k
Although we do not need test set, we require to generate an empty test.txt file for the DGL-KE app
"""
import argparse
import pandas as pd
import numpy as np
import os

def _kg_info(kg_pd, label=""):
    n_rel = kg_pd['r'].nunique()
    n_entity = kg_pd['h'].append(kg_pd['t']).nunique()
    info = "{}\t#entity: {}, #relation: {}, #triplet: {}".format(label, n_entity, n_rel, kg_pd.shape[0])
    return info
def split_kg4KGE(kg_pd, kg_val_ratio, kg_test_ratio=0.0):
    """
    The function for spliting the KG for the dgl/apps/kg repo
    Always assert the training set contains all entities and relations

    Parameters
    ----------
    kg_pd: the whole pandas [h,r t] kg
    kg_test_ratio: the ratio for test set
    kg_val_ratio: the ratio for valid set

    Returns
    -------

    """
    ### analyze KG
    num_triple = kg_pd.shape[0]
    n_entity = kg_pd['h'].append(kg_pd['t']).nunique()
    n_rel = kg_pd['r'].nunique()
    num_test = int(num_triple * kg_test_ratio)
    num_val = int((num_triple - num_test) * kg_val_ratio)
    while True:
        shuffle_idx = np.random.permutation(num_triple)
        train_kg_pd = kg_pd.iloc[shuffle_idx[num_test+num_val: ]]
        if train_kg_pd['h'].append(train_kg_pd['t']).nunique() == n_entity and \
                train_kg_pd['r'].nunique() == n_rel:
            break
    test_kg_pd = kg_pd.iloc[shuffle_idx[:num_test]]
    val_kg_pd = kg_pd.iloc[shuffle_idx[num_test: num_test + num_val]]

    return train_kg_pd, val_kg_pd, test_kg_pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split the entire dataset into train, valid, test sets")
    parser.add_argument('--data_dir', nargs='?',
                        help='the data dir name')
    parser.add_argument('--kg_file', type=str, default='kg.txt',
                        help='the input kg txt file name')
    parser.add_argument('--kg_val_ratio', type=float, default=0.05,
                        help='the validation ratio')
    parser.add_argument('--kg_test_ratio', type=float, default=0.0,
                        help='the validation ratio')
    args = parser.parse_args()
    assert args.data_dir is not None

    _SEP = "\t"
    kg_pd = pd.read_csv(os.path.join(args.data_dir, args.kg_file), sep=_SEP, names=["h", "r", "t"], dtype=str)
    train_kg_pd, val_kg_pd, test_kg_pd = split_kg4KGE(kg_pd, args.kg_val_ratio, args.kg_test_ratio)

    data_l = [train_kg_pd, val_kg_pd, test_kg_pd]
    kg_name = ["train", "valid", "test"]
    info = "{}: val {}, test {}\n".format(args.data_dir, args.kg_val_ratio, args.kg_test_ratio)
    for data, file_label in zip(data_l, kg_name):
        info += _kg_info(data, file_label) + "\n"
        file_name = os.path.join(args.data_dir, "{}.txt".format(file_label))
        data.to_csv(file_name, sep="\t", index=False, header=False)
    f = open(os.path.join(args.data_dir, "log"), "w")
    f.write(info)
    print(info)
    f.close()
