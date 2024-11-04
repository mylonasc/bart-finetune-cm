# interview_nlp_case
This repository contains code for training the `BART` model in Twitter Sentiments.

Some basic cleaning of the data is performed before training.

## Results:


| Dataset | Model | Training setup | F1 | Accuracy | Recall ($\frac{TP}{TP+FN}$) | Precision ($\frac{TP}{TP+FP}$) | 
|---------|-------|----------------|----|----------|-----------------------------|---------------------------------|
|10% test set, seed=`1337` | BART Fine Tuned | 5 epochs, `learning_rate = 3e-5`| **89.58%**| **89.67%** | **88.95%** | **90.22%** |
|10% test set, seed=`1337` | BART Zero Shot (adaptation of prov. script) - p_thresh = 0.20 | NA | 65.57%  | 65.15% | 66.44% | 64.71% | 
|10% test set, seed=`1337` | BART Zero Shot (adaptation of prov. script) - p_thresh = 0.50 | NA | 40.91%  | 59.91% | 27.79% | 77.53% | 



## Stages
The different stages for training a model are as follows:

| Stage | description | result | 
|-------|-------------|--------|
| data cleaning | Clean the data and inspect data for possible noise | **(1)** It was found that all the negative labels (`is_positive = 0`) were before the 800k row, and all `==1` after that. Therefore shuffling the data is necessary before batching. **(2)** It was found that there were non-unique ID pairs, and the label of each pair was reversed. This was remediated by simply taking the first label. **(3)** Some special characters (possibly non-English posts) were in the dataset. These were filtered out due to their small number (less that 1k examples). |
| Stratified Train/test split.  | For model selection it is necessary to contrive a representative validation set. This is done by considering a combination of prediction labels and length of the sentence. | A 90/10 train/test split was created, stratified by label and sentence length. Buckets of sentence lengths were contrived by considering the quantiles of the length distribution. The quantiles of the length distributions were pre-computed using the tokenizer before the train-test split, since they are useful for efficient batching (to minimize padding). |
| Training hyper-parameter selection | Before the main training run, it is useful to have estimates of the performance of different training parameters and model hyper-parameters on training behavior. Full validation becomes too expensive on the full run, and it is useful to pre-estimate these parameters before the run on the full train/test dataset. | Several runs with a representative dataset consisting of 1% of the full dataset were performed, to find a learning rate. A learning rate of 1e-5 was selected out of this procedure, to balance convergence speed (2-3 epochs for best model) and training stability.| 
| Full training run (with train/test split) | Training with the full dataset (train/test split 90/10) to get the final model. | The best model in validation score was in the second epoch. The best test-set binary cross-entropy score achieved was `0.264`.|
| 
