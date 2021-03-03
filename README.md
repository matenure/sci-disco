# Scientific Discovery IBM2

This code is directly revised from ogbl-citation2 (https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/citation2). The code contains self-implement dataset, model and training process. 

## Dataset 

The ogbl-citation2 dataset is a directed graph, representing the citation network between a subset of papers extracted from MAG (Microsoft academic graph). Each node is a paper with 128-dimensional word2vec features that summarizes its title and abstract, and each directed edge indicates that one paper cites another. 

The ogbl-citation2 dataset can be downloaded from https://snap.stanford.edu/ogb/data/linkproppred/citation-v2.zip. Create a directory called "dataset". Unzip the file and place it under the /root/dataset/.
```
unzip citation-v2 -d dataset/
```

This dataset has edge, word2vec feature and year information.
```
Data(edge_index=[2, 30387995], node_year=[2927963, 1], x=[2927963, 128])
```

In citation-v2/raw, edge contains the citation relation between two papers (node). node_year contains the year of paper published and node-feat contains 128 dimension feature for each paper. 

```
# edge.csv example
837791,102
837791,1074631
837791,1658157
837791,2329178
837791,1554695

# node_year.csv example
2016
2007
2006
2002
1965

# node-feature.csv example
-0.045912,-0.132151,-0.022433,-0.13137,0.057672,-0.147349,-0.408782,-0.414552,-0.018326,0.067803,-0.029306,-0.542141,0.171399,-0.002242,-0.206907,0.136606,0.115729,0.404645,0.248616,-0.402928,-0.426349,0.024573,-0.604244,0.030784,-0.157873,0.524152,0.088978,-0.393639,-0.217939,-0.02862,-0.281972,-0.19765,0.086159,0.24011,-0.007625,-0.456285,-0.035679,-0.084924,0.091357,0.165819,0.049626,0.22321,0.283971,0.0361,0.003167,0.160154,0.453907,-0.037929,0.15934,0.075829,0.378658,0.380412,-0.116446,-0.302145,-0.106171,-0.073962,0.058908,-0.277093,-0.010072,0.02238,0.128161,-0.205119,0.102597,0.136588,0.201596,0.049774,-0.116455,0.052415,0.457149,-0.206778,-0.144889,0.061341,0.278827,-0.601361,0.102009,-0.273232,0.077394,0.007043,0.056838,0.159462,-0.022802,0.038153,0.422473,-0.419031,0.107145,0.227556,-0.199461,0.309355,0.84225,0.07739,0.348814,0.030977,0.242354,-0.177634,-0.092405,0.111238,-0.194379,0.07722,0.137518,-0.016501,-0.314134,0.797215,-0.249063,-0.187401,0.108624,0.194635,0.100227,0.127072,-0.04794,0.081264,-0.430801,-0.105431,-0.036654,0.160742,0.382117,0.080911,0.218304,0.041631,0.164972,-0.302996,-0.18883,-0.229541,0.115671,-0.239832,0.037566,-0.010129,0.011574,-0.401273
-0.101635,-0.199759,-0.071075,-0.200401,-0.024454,-0.059139,-0.207714,-0.276148,-0.122317,0.186963,-0.125493,-0.469269,0.075597,0.178241,-0.280112,-0.009582,0.14324,0.172974,0.214746,-0.454254,-0.431507,0.111782,-0.40951,0.015444,0.12723,0.511373,0.010537,-0.393149,-0.291198,0.013478,-0.227882,-0.33149,0.110768,0.155638,0.187104,-0.45447,-0.069795,0.012327,0.177607,0.226875,0.083095,0.199383,0.301913,0.086732,0.014111,0.299076,0.405646,0.087978,0.394384,0.337433,0.45761,0.316297,-0.159394,-0.309778,-0.025395,0.094475,-0.070276,0.021594,0.143423,-0.208096,0.349111,-0.200469,-0.116833,0.103009,0.256844,0.013612,-0.178708,-0.005631,0.464597,-0.167193,-0.167153,-0.001948,0.372564,-0.524499,-0.085119,-0.309263,-0.030215,0.020988,-0.134438,0.183071,-0.036979,0.051214,0.599335,-0.504123,-0.052932,0.224722,-0.122018,0.27477,0.928362,0.107967,0.434697,-0.214554,0.135348,-0.291281,-0.087283,0.25202,-0.224141,-0.016279,0.099986,-0.12937,-0.399735,0.904225,-0.157795,0.027236,0.046097,-0.01698,-0.021966,0.097468,-0.214259,-0.043609,-0.536244,-0.126465,0.032069,0.216483,0.223119,0.377865,0.386312,0.194793,0.212084,-0.306509,-0.080057,-0.357702,0.104477,-0.159406,-0.106419,-0.109911,0.113,-0.325842

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
# result (run twice, each 6 epoches)
Run: 01, Epoch: 01, Loss: 0.5451
Run: 01, Epoch: 02, Loss: 0.2046
Run: 01, Epoch: 03, Loss: 0.1638
Run: 01, Epoch: 04, Loss: 0.1439
Run: 01, Epoch: 05, Loss: 0.1327
Run: 01, Epoch: 06, Loss: 0.1216
Evaluating full-batch GNN on CPU...
Run: 01, Epoch: 06, Loss: 0.1216, Train: 0.7051, Valid: 0.6286, Test: 0.6278
ClusterGCN
Run 01:
Highest Train: 0.7051
Highest Valid: 0.6286
Final Train: 0.7051
Final Test: 0.6278
Run: 02, Epoch: 01, Loss: 0.5384
Run: 02, Epoch: 02, Loss: 0.2065
Run: 02, Epoch: 03, Loss: 0.1630
Run: 02, Epoch: 04, Loss: 0.1466
Run: 02, Epoch: 05, Loss: 0.1333
Run: 02, Epoch: 06, Loss: 0.1210
Evaluating full-batch GNN on CPU...
Run: 02, Epoch: 06, Loss: 0.1210, Train: 0.7006, Valid: 0.6250, Test: 0.6258
ClusterGCN
Run 02:
Highest Train: 0.7006
Highest Valid: 0.6250
Final Train: 0.7006
Final Test: 0.6258
ClusterGCN
tensor([[0.7051, 0.6286, 0.7051, 0.6278],
[0.7006, 0.6250, 0.7006, 0.6258]])
All runs:
Highest Train: 0.7028 ± 0.0032
Highest Valid: 0.6268 ± 0.0025
Final Train: 0.7028 ± 0.0032
Final Test: 0.6268 ± 0.0014
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

