# Divide-and-Conquer Predictive Coding: a structured Bayesian inference algorithm

This repository is the official implementation of [Divide-and-Conquer Predictive Coding](https://arxiv.org/abs/2408.05834).

![Divide-and-conquer PC approximates the joint posterior with bottom-up and recurrent errors.](/figures/dcpc_structure.png "DCPC's overall structure")

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train any experiment in the paper, run this command:

```train
python train.py -c experiments/<experiment_name>_config.json
```

As an example, you can train the Deep Latent Gaussian Model on MNIST via
```train_mnist
python train.py -c experiments/dcpc_mnist_config.json
```
and then watch training progress on Tensorboard via
```tensorboard_mnist
tensorboard --logdir saved/log/Mnist_Dcpc/<date_and_time>
```
The JSON configuration files contain all necessary hyperparameters.

## Evaluation

To evaluate any experiment in the paper, open:
```eval_jupyter
experiments/<experiment_name>_eval.ipynb
```
in Jupyter Lab, making sure to set `experiments/` as your current directory. You
can also run an evaluation notebook from the command line:
```eval_nbconvert
cd dcpc_paper/experiments;
jupyter nbconvert --execute --to notebook --inplace <experiment_name>_eval.ipynb
```

The following specific notebooks, once equipped with the pre-trained weights and
particles documented below, will reproduce the figures in the paper:

  * Figure 3a comes from `experiments/dcpc_mnist_eval.ipynb` as `dcpc_mnist_recons.pdf`;
  * Figure 3b comes from `experiments/dcpc_emnist_eval.ipynb` as `dcpc_emnist_recons.pdf`;
  * Figure 3c comes from `experiments/dcpc_fashionmnist_eval.ipynb` as `dcpc_fashionmnist_recons.pdf`;
  * Figure 4a comes from `experiments/dcpc_celeba_eval.ipynb` as `dcpc_celeba_recons.pdf`; and
  * Figure 4b comes from `experiments/dcpc_celeba_eval.ipynb` as `dcpc_celeba_predictive.pdf`.

## Pre-trained Weights and Particles

You can download the pretrained weights and particles corresponding to the paper
[here](https://drive.google.com/drive/folders/1yK52Geb6hk3F947Ls4tS0pkY2L5RpKew?usp=drive_link).
The following specific files correspond to the notebooks listed above for reproducing
the paper's major figures:

- Weights and particles for the [Deep Latent Gaussian Model](https://drive.google.com/file/d/1fNQcsx-yINFv1LNHHeKNHyU3TETkev8R/view?usp=share_link) (Figure 3a) trained on MNIST;
- Weights and particles for the [Deep Latent Gaussian Model](https://drive.google.com/file/d/1GV5a8evIIJmV5816zQRHwTSl2vmx91zR/view?usp=share_link) (Figure 3b) trained on Extended MNIST;
- Weights and particles for the [Deep Latent Gaussian Model](https://drive.google.com/file/d/17n2-TzTmOMH8dbnYnkI_SM2i5aubqmmk/view?usp=share_link) (Figure 3c) trained on Fashion MNIST; and
- Weights and particles for the [Convolutional Generator network](https://drive.google.com/file/d/11mFxrefTcE0k9hIyyu5r2Mg1HVvZzRei/view?usp=share_link) (Figure 4) trained on CelebA.

## Results

Quantitatively, we evaluated DCPC as a Bayesian inference algorithm in generative
modeling, via the Frechet Inception Distance (FID) score on CelebA. We
achieve the following results in comparison to other inference algorithms,
holding the model architectures constant:

### Training time and Frechet Inception Distances
| Inference algorithm        | Likelihood          | Resolution | Sweeps x Epochs | FID  |
| -------------------------- | ------------------- | ---------- | --------------- | ---- |
| Particle Gradient Descent  | Normal              |   32 x 32  |      1 x 100    | 100  |
| **DCPC (ours)**            | Normal              |   32 x 32  |      1 x 100    | **82.7** |
| Langevin Predictive Coding | Discretized Normal  |   64 x 64  |    300 x 15     | 120  |
| Variational Autoencoder    | Discretized Normal  |   64 x 64  |      1 x 4500   | 86.3 |
| **DCPC (ours)**            | Discretized Normal  |   64 x 64  |     30 x 150    | **79.0** |

For those wondering what a "discretized Normal" distribution is, we provide an
implementation as a Pyro distribution, in `utils.util.DiscretizedGaussian`. The
definition comes from the literature on diffusion models.

## Contributing

Feel free to contribute back to this code or fork it, as long as you remain in compliance with the following license:

MIT License

Copyright (c) 2023-2024 Eli Sennesh, Hao Wu, and Tommaso Salvatori

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
