# Cleanup Nodes
\<Add short project description here\>

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
