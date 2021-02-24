# Scientific Discovery IBM2

This code is directly revised from ogbl-citation2 (https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/citation2). The code contains self-implement dataset, model and training process. But no test and validation part. 

## Dataset 

The ogbl-citation2 dataset is a directed graph, representing the citation network between a subset of papers extracted from MAG (Microsoft academic graph). Each node is a paper with 128-dimensional word2vec features that summarizes its title and abstract, and each directed edge indicates that one paper cites another. 

The ogbl-citation2 dataset can be downloaded from https://snap.stanford.edu/ogb/data/linkproppred/citation-v2.zip. Create a directory call "dataset". Unzip the file and place it under the /root/dataset/.
```
unzip citation-v2 -d dataset/
```

This dataset has edge, word2vec feature and year information.
```
Data(edge_index=[2, 30387995], node_year=[2927963, 1], x=[2927963, 128])
```

## Model 

Two models, one is called GCN which takes the node as input and output the feature vector. The other is called LinkPredictor which output the citation prossibility. This simple model only contains three layers.
```
GCN(
(convs): ModuleList(
(0): GCNConv(128, 128)
(1): GCNConv(128, 128)
(2): GCNConv(128, 128)
)
)
LinkPredictor(
(lins): ModuleList(
(0): Linear(in_features=128, out_features=128, bias=True)
(1): Linear(in_features=128, out_features=128, bias=True)
(2): Linear(in_features=128, out_features=1, bias=True)
)
)
```

## Train

Run the train.py to start processed the data and start training

```
> python train.py
```

```
Run: 01, Epoch: 01, Loss: 0.4359
Run: 01, Epoch: 02, Loss: 0.2783
Run: 01, Epoch: 03, Loss: 0.2514
Run: 01, Epoch: 04, Loss: 0.2464
Run: 01, Epoch: 05, Loss: 0.2320
Run: 01, Epoch: 06, Loss: 0.2263
Run: 01, Epoch: 07, Loss: 0.2290
Run: 01, Epoch: 08, Loss: 0.2149
Run: 01, Epoch: 09, Loss: 0.2139
Run: 01, Epoch: 10, Loss: 0.2121
ClusterGCN
Run: 02, Epoch: 01, Loss: 0.4306
Run: 02, Epoch: 02, Loss: 0.2744
Run: 02, Epoch: 03, Loss: 0.2468
Run: 02, Epoch: 04, Loss: 0.2348
Run: 02, Epoch: 05, Loss: 0.2277
Run: 02, Epoch: 06, Loss: 0.2297
Run: 02, Epoch: 07, Loss: 0.2144
Run: 02, Epoch: 08, Loss: 0.2158
Run: 02, Epoch: 09, Loss: 0.2105
Run: 02, Epoch: 10, Loss: 0.2145
...
```

## Environment

```
conda create -name 696ds python=3.7.0
conda activate 696ds
conda install -c pytorch pytorch=1.7.0  # install pytorch
conda install -c anaconda cudatoolkit=10.2  # install cuda

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-geometric

pip install ogb # install ogb
```

