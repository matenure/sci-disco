# Scientific Discovery IBM2

Run train.py for scientific discovery project using link prediction model
```
python train.py 
```

Output
```
Namespace(batch_size=128, device=0, dropout=0.8, epochs=50, eval_steps=1, hidden_channels=128, log_steps=1, lr=0.0001, num_layers=3, num_workers=0, runs=1)
cpu
Data(edge_index=[2, 2277], x=[2015, 512])
GCN(
(convs): ModuleList(
(0): GCNConv(512, 128)
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
Run: 01, Epoch: 01, Loss: 1.3877
Evaluating full-batch GNN on CPU...
Run: 01 ,
Epoch: 01 ,
Loss: 1.3877 ,
Train_Precision: 0.0600 ,
Valid_Precision: 0.1100 ,
Test_Precision: 0.1500
Run: 01, Epoch: 02, Loss: 1.3870
Evaluating full-batch GNN on CPU...
Run: 01 ,
Epoch: 02 ,
Loss: 1.3870 ,
Train_Precision: 0.1600 ,
Valid_Precision: 0.1400 ,
Test_Precision: 0.1700
Run: 01, Epoch: 03, Loss: 1.3874
Evaluating full-batch GNN on CPU...
Run: 01 ,
Epoch: 03 ,
Loss: 1.3874 ,
Train_Precision: 0.2300 ,
Valid_Precision: 0.1400 ,
Test_Precision: 0.2000
Run: 01, Epoch: 04, Loss: 1.3866
Evaluating full-batch GNN on CPU...
Run: 01 ,
Epoch: 04 ,
Loss: 1.3866 ,
Train_Precision: 0.2700 ,
Valid_Precision: 0.1400 ,
Test_Precision: 0.2100
Run: 01, Epoch: 05, Loss: 1.3865
Evaluating full-batch GNN on CPU...
Run: 01 ,
Epoch: 05 ,
Loss: 1.3865 ,
Train_Precision: 0.2500 ,
Valid_Precision: 0.1500 ,
Test_Precision: 0.2200
Run: 01, Epoch: 06, Loss: 1.3872
Evaluating full-batch GNN on CPU...
...
```
