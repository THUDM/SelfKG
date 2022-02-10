# SelfKG

SelfKG: Self-Supervised Entity Alignment in Knowledge Graphs.   <img src="img/combine.PNG" style="zoom:100%;" />

The paper of SelfKG is accepted by The Web Conference 2022!

## 1. Setup

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

You can download the our data from [here](https://drive.google.com/drive/folders/1vuXC6A0WETEr-b2yA6Y1ZxR8Dsli4xLr?usp=sharing), and the final structure our project should be:

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

You can use 

```bash experiments.sh```

 to reproduce our experiments results. For more details, please refer to `experiments.sh` and our code.
