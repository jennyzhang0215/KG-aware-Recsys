# KG-aware-Recsys

This tutorial is about how to leverage rich item attributes as an item-based knowledge graph (KG) for the recommender system (recsys). 
We utilize the [DGL-KE](https://github.com/dmlc/dgl/tree/master/apps/kg) to train the knowledge graph and 
then use the pretrained item embeddings as item features in a recsys model.

The training process is not end-to-end in order to make use of the high performance of DGL-KE.

1. Pretrain the item-KG graph using DGL-KE:
  ```
  git clone https://github.com/dmlc/dgl.git
  cd dgl/apps/kg
  ```
  
  - We recommend you to format the KE data as `format 1` in the README page and split 5% of the KG triplets as validation set. No testing set is required.

  > Data Format 1:
  > - **entities.dict** contains pairs of (entity Id, entity name). The number of rows is the number of entities (nodes).
  > - **relations.dict** contains pairs of (relation Id, relation name). The number of rows is the number of relations.
  > - **train.txt** stores edges in the training set. They are stored as triples of (head, rel, tail).
  > - **valid.txt** stores edges in the validation set. They are stored as triples of (head, rel, tail).
  - In a KG, we call nodes as an *entities* and edges as a *relations*. So there are two kinds of entities in KG, *item entities* and *non-item entities*.
  - When you map entities to entity ids in **entities.dict** file, please guarantee that the item entities has smaller ids. 
  For example, movie items *Titanic* and *Star Wars* has smaller ids 0 and 1, while attribute entities like genre value *Action* and Actor value *Leonardo DiCaprio* has bigger ids 2 and 3, respectively.
  
  ```
  0 Titanic
  1 Star Wars
  2 Action
  3 Leonardo DiCaprio
  ```
  - When remapping entities Please guarantee that the item entities place place first 
  