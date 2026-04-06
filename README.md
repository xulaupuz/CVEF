# Dual CNN and ViT Experts Fusion for Open Set Recognition
### Run the project
Download datasets, then modify the dataset path in `/datasets/osrload.py`, then run:
```
python osr_main.py
python ood_main.py
python acc_main.py
```
### Requirements
- pytorch
- easydict
- numpy
- Pillow
- PyYAML
- scikit_learn

### Special Thanks
This project mainly focuses on improvements to [MEDAF](https://github.com/Vanixxz/MEDAF). I **strongly** recommend to use his project to understand the MoE structure of this work. Our code is also a modification on his project. We cited this work in our article.

### Cite Us
[1] Ding K, Mao Y, Chen H, et al. Dual CNN and ViT Experts Fusion for Open Set Recognition[J]. Neural Networks, 2026: 108910.
```
@article{DING2026108910,
title = {Dual CNN and ViT experts fusion for open set recognition},
journal = {Neural Networks},
volume = {201},
pages = {108910},
year = {2026},
issn = {0893-6080},
doi = {10.1016/j.neunet.2026.108910},
url = {https://www.sciencedirect.com/science/article/pii/S0893608026003710},
author = {Kai Ding and Yu Mao and Hui Chen and Yaojin Lin},
keywords = {Classification, Open set recognition, Vision transformer, Mixture of experts},
}
```
