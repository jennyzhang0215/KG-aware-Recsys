# KG-aware-Recsys

This tutorial is about how to leverage rich item attributes as an item-based knowledge graph (KG) for the recommender system (recsys). 
We utilize the [DGL-KE](https://github.com/dmlc/dgl/tree/master/apps/kg) to train the knowledge graph and 
then use the pretrained item embeddings as item features in a recsys model (PinSAGE/GraphSAGE).

The training process is not end-to-end in order to make use of the high performance of DGL-KE.

### 1. Prepare the item-KG data for DGL-KE:
  ```
  git clone --recursive https://github.com/dmlc/dgl.git
  cd dgl/apps/kg
  mkdir data
  ```
  - Please read the README of DGL-KE first. It is easy to follow.
  - We recommend you to format the KG data as `format 1` in the README page and split 5% of the KG triplets as a validation set. No testing set is required.

  > Data Format 1:
  > - **entities.dict** contains pairs of (entity Id, entity name). The number of rows is the number of entities (nodes).
  > - **relations.dict** contains pairs of (relation Id, relation name). The number of rows is the number of relations.
  > - **train.txt** stores edges in the training set. They are stored as triples of (head, rel, tail).
  > - **valid.txt** stores edges in the validation set. They are stored as triples of (head, rel, tail).
  > - **test.txt** stores edges in the test set. They are stored as triples of (head, rel, tail).

  - In a KG, we call nodes as *entities* and edges as *relations*. So there are two kinds of entities in KG, *item entities* and *non-item entities*.
  - When you map entities to entity ids in the **entities.dict** file, please guarantee that the item entities has smaller ids for use our provided scripts. 
  For example, movie items *Titanic* and *Star Wars* has smaller ids 0 and 1, while attribute entities like genre *Action* and actor *Leonardo DiCaprio* has bigger ids 2 and 3, respectively.
  
  ```
  0 Titanic
  1 Star Wars
  2 Action
  3 Leonardo DiCaprio
  ```
  - (Optinal) You can use the `kge_utils/split_valid.py` script to split a train and valid set 
  as well as an empty test set, which guarantees that all the entities and relations are seen in the training set. 
  For more details, please read the `split_valid.py` file.

### 2. Run a KGE model in DGL-KE 
  - Copy the prepared dataset to the DGL-KE `data` folder. 
  All the data should be in a directory and you should copy the entire directory to the `data` folder.
  ```bash
  cp -r [path of your prepared dataset] [dgl/apps/kg/data/.]
  ```
  - Run a training script in DGL-KE. Remember to save the learned embeddings by providing the path with `--save_emb` when running
`train.py`. `[dataset]` indicates the name of the data directory. 
  ```bash
  cd dgl/apps/kg
  DGLBACKEND=pytorch python3 train.py --model [model] --dataset [dataset] --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 100000 \
    --batch_size_eval 16 --gpu 0 --valid -adv --save_emb emb
  ```
You may change different KGE models and hyperparameters and choose the learned embeddings with the best validation score.
The learned entity embeddings (including item entities and non-item entities) are stored in the `[save_emb]/[dataset]_[model]_entity.npy` file.

### 3. Extract the learned entitiy embeddings as item features
  - Copy the `[save_emb]/[dataset]_[model]_entity.npy` file in the DGL-KE into this repo's `gen_feature/KG_trained_embed' folder
  - Run the `gen_feature/convert_entityEmb2fea.py` script to generate the feature pickle file. 
  The `n_item` argument is required and defined by you own dataset. 
  ```bash
  cd gen_feature
  cp dgl/apps/kg/[save_emb]/[dataset]_[model]_entity.npy KG_trained_embed/.
  python convert_entityEmb2fea.py --n_item [n_item]
  ```
  
  Now the generated `[data_name]_entity_embed_features.pkl` file can be used as the input file in a recsys model, e.g., GraphSAGE/PinDAGE.


 