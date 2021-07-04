# SelfKG
Codes for ``A Self-supervised Method for Entity Alignment''
## 1.Get Started
**Dependencies**
```
torch 1.9.0
bson 0.5.10
faiss-cpu 1.7.1
numpy 1.20.2
pandas 1.2.5
tqdm 4.61.1
transformers 4.8.2
```
**Download Data**
You can download the prepared data [here](https://cloud.tsinghua.edu.cn/f/9c5dfe2f4b064c998ef6/?dl=1).

**Run**
Run `python run_{}.py` to reproduce our experiments (e.g. `run_LaBSE_neighbor.py` for SelfKG with LaBSE and neighbor informations enabled on DBP15K).