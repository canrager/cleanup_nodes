# Cleanup Nodes

[Blogpost](https://www.lesswrong.com/posts/2PucFqdRyEvaHb4Hn/an-adversarial-example-for-direct-logit-attribution-memory)

We provide concrete evidence for memory management or clean-up in a 4-layer transformer [gelu-4l](http://neelnanda.io/toy-models). Then we examine implications on  Direct Logit Attribution (DLA), a rough method to measure the relevance of attention heads and MLP layers w.r.t. a specific task. We conclude DLA is misleading because it does not account for the clean-up.

James Dao, Yeu-Tong Lau, Can Rager, and Jett Janiak did this work as the final capstone project of ARENA in 2023. Alignment Research Engineering Acellerator (ARENA) is a fellowship covering software engineering, natural language processing, reinforcement learning and distributed computing.

## Environment Setup

If you don't already have conda/mamba, install it with:

```bash
cd ~ && \
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh && \
bash Mambaforge-Linux-x86_64.sh -b && \
mambaforge/bin/mamba init bash && \
exec bash
```

Clone the repo and `cd` into it:

```bash
git clone https://github.com/canrager/cleanup_nodes.git && cd cleanup_nodes
```

Install the conda/mamba environment and activate it:

```bash
mamba env create -f environment.yaml && mamba activate cleanup
```
