# NIID-Bench
This is the code of paper 《一种面向智联网的高效联邦学习》.


This code runs a benchmark for federated learning algorithms under non-IID data distribution scenarios. Specifically, we implement 6 federated learning algorithms (FedAvg, FedProx, fedavgm,fedadam & Fedcnm).





## Usage
Here is one example to run this code:
```
python experiments.py --model=resnet \
    --dataset=cifar10 \
    --alg=fedprox \
    --lr=0.01 \
    --batch-size=32 \
    --epochs=1 \
    --n_parties=100 \
    --mu=0.01 \
    --rho=0.9 \
    --comm_round=500 \
    --partition=noniid-labeldir \
    --beta=0.5\
    --device='cuda:0'\
    --datadir='./data/' \
    --logdir='./logs/' \
    --sample=0.1 \
    --init_seed=0
```

| Parameter                      | Description                                                                                                                                                                                              |
| ----------------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model` | The model architecture. Options: `simple-cnn`, `vgg`, `resnet`, `mlp`,`resnet`. Default = `resnet`.                                                                                                      |
| `dataset`      | Dataset to use. Options: `mnist`, `cifar10`,`cifar100`, `fmnist`, `svhn`, `generated`, `femnist`, `a9a`, `rcv1`, `covtype`. Default = `cifar10`.                                                         |
| `alg` | The training algorithm. Options: `fedavg`, `fedprox`, `scaffold`, `fednova`. Default = `fedavg`.                                                                                                         |
| `lr` | Learning rate for the local models, default = `0.01`.                                                                                                                                                    |
| `batch-size` | Batch size, default = `32`.                                                                                                                                                                              |
| `epochs` | Number of local training epochs, default = `1`.                                                                                                                                                          |
| `n_parties` | Number of parties, default = `100`.                                                                                                                                                                      |
| `mu` | The proximal term parameter for FedProx, default = `0.001`.                                                                                                                                              |
| `rho` | The parameter controlling the momentum SGD, default = `0`.                                                                                                                                               |
| `comm_round`    | Number of communication rounds to use, default = `500`.                                                                                                                                                  |
| `partition`    | The partition way. Options: `homo`, `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns), `real`, `iid-diff-quantity`. Default = `noniid-labeldir` |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition, default = `0.3`.                                                                                                  |
| `device` | Specify the device to run the program, default = `cuda:0`.                                                                                                                                               |
| `datadir` | The path of the dataset, default = `./data/`.                                                                                                                                                            |
| `logdir` | The path to store the logs, default = `./logs/`.                                                                                                                                                         |
| `sample` | Ratio of parties that participate in each communication round, default = `0.1`.                                                                                                                          |
| `init_seed` | The initial seed, default = `0`.                                                                                                                                                                         |
