<img src="img/combine.PNG" style="zoom:100%;" />

<p align="center"><a href="https://github.com/THUDM/SelfKG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/THUDM/SelfKG" /></a>

# SelfKG: Self-Supervised Entity Alignment in Knowledge Graphs

Original implementation for paper SelfKG: Self-Supervised Entity Alignment in Knowledge Graphs.   

This paper is accepted by  [The Web Conference2022](https://www2022.thewebconf.org/)! :satisfied:

SelfKG is the **first** **self-supervised** entity alignment method **without label supervision**, which can **match or achieve comparable results with state-of-the-art supervised baselines**. The performance of SelfKG suggests self-supervised learning offers great potential for entity alignment in Knowledge Graphs.

[SelfKG: Self-Supervised Entity Alignment in Knowledge Graphs](#SelfKG: Self-Supervised Entity Alignment in Knowledge Graphs)

- [Installation](#installation)
  - [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Data Preparation](#data-preparation)
  - [Run Experiments](#run-experiments)
- [❗ Common Issues](#-common-issues)
- [Citing SelfKG](#citing-selfkg)

## Installation

### Requirements

```txt
torch==1.9.0
faiss-cpu==1.7.1
numpy>=1.21
pandas==1.2.5
tqdm==4.61.1
transformers==4.8.2
```

You can use [`setup.sh`](https://github.com/THUDM/SelfKG/blob/main/setup.sh) to set up your Anaconda environment.

## Quick Start

### Data Preparation

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

### Run Experiments

You can use

```bash experiments.sh```

 to reproduce our experiments results. For more details, please refer to [`experiments.sh`](https://github.com/THUDM/SelfKG/blob/main/experiments.sh) and our code.

## ❗ Common Issues

<details>
<summary>
"XXX file not found"
</summary>
<br/>
Please make sure you've downloaded all the dataset according to README.md
</details>

to be continued ...


## Citing SelfKG

If you use SelfKGin your research or wish to refer to the baseline results, please use the following BibTeX.

```
in preparation ...
```
