Sentence_classification project aims to create python module with easy to use Tensorflow based classifiers for text classification on the level of single line.<br>
Enron manually annotated database was used as a base for all evaluations.<br>
Two approaches of classification are implemented now:<br>
- **MovingWindowSentenceClassifier** - sliding window, with odd vertical dimension - classifier bsaed on CNN/BiGRU layers, 
> - self-created embeddings version [summary](https://github.com/marcinchomicz/sentence_classification/blob/dev/optim/moving_window_clf_optim_top_results/summary_kindly-penguin-973.txt), [diagram](https://github.com/marcinchomicz/sentence_classification/blob/dev/optim/moving_window_clf_optim_top_results/diagram_kindly-penguin-973.png)
- **ContextBranchSentenceClassifier** - context branches, with preceeding and following lines passed to separated branches, 
> - self-created embeddings version [summary](https://github.com/marcinchomicz/sentence_classification/blob/dev/optim/context_branch_clf_optim_top_results/summary_gaudy-deer-889.txt), [diagram](https://github.com/marcinchomicz/sentence_classification/blob/dev/optim/context_branch_clf_optim_top_results/diagram_gaudy-deer-889.png)

Directories content:
- **classes** - ready to use classifiers
- **dataset** - enron files annotated
- **examples** - example scripts and Jupyter notebooks
- **optim** - results of crossvalidated hyperparameters optimization using [Neural Network Intelligence](https://www.microsoft.com/en-us/research/project/neural-network-intelligence/)
- **test** - unittests
