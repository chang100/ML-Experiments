# NiN Model

Network in Network model based on https://arxiv.org/pdf/1312.4400.pdf

# Architecture

| Layer Type | Parameters |
|------------|----------------------------|
| Conv       |  Kernel: 5x5, Channel: 192 |
| ReLU       |                            |
| Conv       |  Kernel: 1x1, Channel: 160 |
| ReLU       |                            |
| Conv       |  Kernel: 1x1, Channel: 96  |
| ReLU       |                            |
| Maxpool    |  Kernel: 3x3, Stride: 2    |
| Conv       |  Kernel: 5x5, Channel: 192 |
| ReLU       |                            |
| Conv       |  Kernel: 1x1, Channel: 192 |
| ReLU       |                            |
| Conv       |  Kernel: 1x1, Channel: 192 |
| ReLU       |                            |
| Avgpool    |  Kernel: 3x3, Stride: 2    |
| Conv       |  Kernel: 5x5, Channel: 192 |
| ReLU       |                            |
| Conv       |  Kernel: 1x1, Channel: 192 |
| ReLU       |                            |
| Conv       |  Kernel: 1x1, Channel: 10  |
| ReLU       |                            |
| AvgPool    |  Kernel: 7x7, Stride: 1    |

# Results
85.2% validation accuracy on Cifar-10
