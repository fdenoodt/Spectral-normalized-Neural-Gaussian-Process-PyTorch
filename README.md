# Spectral Normalized Neural Gaussian Process (PyTorch Implementation)
PyTorch implementation of SNGP as found in https://arxiv.org/pdf/2006.10108.pdf.

This repo follows the implementation found at https://www.tensorflow.org/tutorials/understanding/sngp but uses PyTorch.
Unlike the original paper and this implementation the notebook also illustrates how the principles of SNGP can be applied to a regression task to estimate uncertainty.

Please note that this has been developed entirely for personal use, however it is freely distributed. It has been made available in case it can be of value to ML practitioners and researchers.

## Development

Set up a conda environment as follows:

```bash
micromamba create -f environment.yml
micromamba activate sngp
```

Run the sngp script:

```bash
python sngp.py
```

Generate and launch the sngp jupyter notebook:

```bash
jupytext --to ipynb sngp.py
jupyter notebook sngp.ipynb
```

Update the sngp markdown file:

```bash
jupyter nbconvert --execute --to markdown sngp.ipynb
```
