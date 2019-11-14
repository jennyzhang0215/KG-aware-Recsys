# KG-aware-Recsys

This tutorial is about how to leverage rich item attributes as an item-based knowledge graph (KG) for the recommender system (recsys). We utilize the [DGL-KE](https://github.com/dmlc/dgl/tree/master/apps/kg) to train the knowledge graph and then the pretrained item embeddings can be used as item features in a recsys model.

The training process is not end-to-end in order to make use of the high performance of DGL-KE.

1. Pretrain the item-KG graph using DGL-KE:
  ```
  git clone DGL: git clone https://github.com/dmlc/dgl.git
  cd dgl/apps/kg
  ```
  - Format the KG data as the input form of DGL-KE by split 5% of the KG triplets as validation set. No testing set is required.
  - We recommend to format the KE data as `format 1` in the README page. Then you can run the script to split a validation set.
  - Please guarantee that the item entities place place first 
  
