Sentence_classification project aims to create python module with easy to use Tensorflow based classifiers for text classification on the level of single line.
Enron manually annotated database was used as a base for all evaluations.
Two approaches of classification are implemented now:
- sliding window, with odd vertical dimension - classifier bsaed on CNN/BiGRU layers,
- context branches, with preceeding and following lines passed to separated branches.

Directories content:
- classes - ready to use classifiers
- dataset - enron files annotated
- examples - example scripts and Jupyter notebooks
- optim - results of crossvalidated hyperparameters optimization using [Neural Network Intelligence](https://www.microsoft.com/en-us/research/project/neural-network-intelligence/)
