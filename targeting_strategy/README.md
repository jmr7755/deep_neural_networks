## Scenario
A financial institution desires to refine its targeting strategy and grow the client population leveraging third party credit data.

## Assumptions, limitations, conclusion
**Assumptions:**
Main assumption is that stakeholder will prefer to have more false positives (e.g. sending advertisement to people that most likely won't become a customer) than to have false negatives (missing potential customers).
Therefore, ROC-AUC was selected as a single performance evaluation metric (for ease of comparison between different models).

**Limitations:**
Main limitation is lack of proper estimate of Bayess Classifier performance. For instance, for tasks like image classification Bayess Classifier error is usually assumed to be zero, as human performance is around 0% error. However, for task like this it's hard to estimate optimal performance, therefore we can't conclude whether our model is too simple (has removable bias), or we are already performing at best possible rate.

Another limiting factor is small number of samples with 1 target. It may be possible to significantly increase model performance by having more data for class 1.

**Conclusion:**
Focal loss allows to train model on skewed data without data augumentation or downsampling.

α  was set to be equal inverse normalized frequency of appearing of this class in data. This can be taken as starting default value that can be tuned later with cross-validation. However, in our case this value appeared to be optimal or close to optimal.
γ  was tuned using validation set.
Focal loss gave better performance than using SMOTE to oversample rare class, or than using downsamlping or combinations of above. It also gave better results than simply using weights to account skewed target.
Setting bias of the final layer to  log(1−π/π) , where  π  is normalized frequency of rare class ensure faster convergence and numerical stability. With SMOTEd and downsampled data local optimas of predicting all 0s were a significant problem.
Tweaking  γ  parameter can lead to either more correctly classified 0 classes, or to more correctly classified 1 classes.

**What else was tried:**
SMOTE,
downsampling,
ensemble,
two stage training (SMOTEd data at stage 1 + downsampled data at stage 2).

**References:**
https://arxiv.org/pdf/1804.07612.pdf
https://www.kaggle.com/abazdyrev/keras-nn-focal-loss-experiments
