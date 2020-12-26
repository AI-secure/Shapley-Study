# Shapley-Study

## Setup

Use the following command to install the package:

```bash
python setup.py install
```

## Overview

We provide 13 datasets, 5 feature extractors, 5 applications, and 6 data valuation measures for a comprehensive evaluation. The choices supported are listed below. 

```
Datasets        mnist, fashionmnist, svhn, cifar, pubfig, tinyimagenet, usps, uci adult
Extractors      VGG11, MobileNet, ResNet18, EfficientNet, Inception-V3
Applications    Noisy label detection, Watermark removal, Data summarization, Active data acquisition, Domain adaptation
Measures        KNN-Shapley, TMC-Shapley, G-Shapley, LOO, KNN-LOO, Random
```

The customized datasets including injected watermarks as well as other preprocessed datasets used in the code can be found on [Google Drive](https://drive.google.com/drive/folders/1vJ4PDyLq9Ud1d5NBeQjZiJHEeWqRXhrk?usp=sharing). You are recommended to download the folder ``Shapley_data`` and put it under the root folder (the same as samples.ipynb) for the purpose of testing.

## Usage

**Step 1.** Apply a certain extractor to a certain dataset to extract the embeddings, implemented in the form of ``extract_embeddings(extractor, dataset)``, e.g.

``
python -m shapley.embedding.extract_embeddings --extractor resnet18 --dataset mnist
``

**Step 2.** Use a certain extracted embedding along with a certain measure in a certain application, implemented in the form of 

```
measure = ...
app = ...
app.run(measure)
```

See ``samples.ipynb`` for the sample testcases.

## Changelog

**2020.12.27**

Add the PyTorch implementation for KNN-Shapley calculation in [shapley/measures/KNN_Shapley.py](https://github.com/AI-secure/Shapley-Study/blob/master/shapley/measures/KNN_Shapley.py). The PyTorch implementation runs faster than the original NumPy implementation since the operations are paralleled.

One standalone experiment can be found in [samples.ipynb](https://github.com/AI-secure/Shapley-Study/blob/master/samples.ipynb).