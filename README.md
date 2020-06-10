## KNN-Shapley

Thie repo is an implementation of "Scalability vs. Utility: Must We Sacriﬁce One for the Other in Shapley-enriched Data Workﬂows?"

### Overview

We provide 13 datasets, 5 feature extractors, 5 applications, and 6 data valuation measures for a comprehensive evaluation. The choices supported are listed below. 

```
Datasets        mnist, fashionmnist, svhn, cifar, pubfig, tinyimagenet, usps, uci adult
Extractors      VGG11, MobileNet, ResNet18, EfficientNet, Inception-V3
Applications    Noisy label detection, Watermark removal, Data summarization, Active data acquisition, Domain adaptation
Measures        KNN-Shapley, TMC-Shapley, G-Shapley, LOO, KNN-LOO, Random
```

### Usage

**Step 1.** Apply a certain extractor to a certain dataset to extract the embeddings, implemented in the form of ``extract_embeddings(extractor, dataset)``, e.g.

``
python -m shapley.embedding.extract_embeddings --extractor resnet18 --dataset mnist
``

**Step 2.** Use a certain extracted embedding along with a certain measure in a certain application, implemented in the form of ``run_experiment(embedding, measure, application)``.
