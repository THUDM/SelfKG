# SelfKG

Codes for ''A Self-supervised Method for Entity Alignment''

## 1. Set up

**Dependencies**

```bash
torch # just choose the version based on your machine 
faiss-cpu
numpy
pandas
tqdm
transformers
```
You can use `setup.sh` to set up your Anaconda environment.

**Data**

You can download the our data from [here](https://cloud.tsinghua.edu.cn/d/c1df705453784e568a23/), and the final structure our project should be:

```bash
├── data
│   ├── DBP15K
│   │   ├── fr_en
│   │   ├── ja_en
│   │   └── zh_en
│   ├── DWY100K
│   │   ├── dbp_wd
│   │   └── dbp_yg
│   └── LaBSE
│       ├── bert_config.json
│       ├── bert_model.ckpt.index
│       ├── checkpoint
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt
├── loader
├── model
├── run_DWY_LaBSE_neighbor.py
├── run_LaBSE_neighbor.py
├── run_LaBSE_SSL_DWY.py
├── run_LaBSE_SSL.py
├── script
│   └── preprocess
└── settings.py
```

## 2. Run experiments

You can use `experiments.sh` to reproduce our experiments results. For more details, please refer to `experiments.sh` and our code.
