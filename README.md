# NLP: Machine Reading Comprehension
## Abstract
In this experiment, we aimed to optimize the performance of a reading comprehension model using different pre-trained architectures, hyperparameters, and optimization strategies. We experimented with [BERT](https://arxiv.org/abs/1810.04805) and ALBERT models of various scales and adjusted hyperparameters, such as learning rate, batch size, and max sequence length. Moreover, we employed optimization strategies like Adam optimizer with weight decay, warmup, and learning rate decay. The experimental results provided insights into the strengths and weaknesses of each model, allowing us to select the most suitable model for the reading comprehension task. Based on our experiments, the ALBERT-large model emerged as the best choice, with an "exact" score of 81.42845110755496 and an "F1" score of 84.86029930097264 on the evaluation set. Through these adjustments and optimization strategies, we achieved good model performance while maintaining stability and generalization during training.

## 1 Introduction

### 1.1 Task Background and Related Work
Reading comprehension is a vital NLP task. Pre-trained language models like BERT and ALBERT have advanced this field significantly. BERT captures bidirectional context but has slow training speeds, while ALBERT maintains performance with fewer parameters and faster training. Other techniques like XLNet and RoBERTa have also achieved top performance. This task explores using these models to enhance reading comprehension.


## 2 Improving Baseline Model Performance


### 2.1 Hyperparameter Tuning

In our experiments, we focused on tuning the following hyperparameters:
+ **Batch size**:We tested different batch sizes, such as 8, 16, and 32, and analyzed the impact on the model's performance.
  
+ **Learning rate (Adam)**:We experimented with three learning rates (5e-05, 3e-05, and 1e-05) and evaluated their effects on the model's performance.

+ **Max sequence length**:This parameter determines the maximum length of input sequences in the BERT model, which is composed of the question and relevant paragraph. We tested max sequence lengths of 384 and 512.

Additionally, we applied optimization strategies such as using a modified Adam optimizer with weight decay, warmup, and learning rate decay.

### 2.2 Neural Network Architectures

We explored different pre-trained model architectures to improve performance on the reading comprehension task. We tested BERT and ALBERT models of various sizes, as well as models fine-tuned on the SQuAD2 dataset.

+ **BERT-SQuAD2**:This model is fine-tuned on the SQuAD2 dataset, specifically designed for reading comprehension tasks. However, we ultimately decided against using this model.

+ **ALBERT (A Lite BERT)**:We tested four different sizes of ALBERT models, including Albert-base, Albert-large, Albert-xlarge, and Albert-xxlarge.

By training and testing on various BERT and ALBERT models, we evaluated their performance on the reading comprehension task.


## 3 Experiment and Analysis

### 3.1 Dataset Analysis
The Stanford Question Answering Dataset (SQuAD) 2.0 expands the original SQuAD dataset, widely used for evaluating question-answering models. It combines the original question-answer pairs with over 50,000 unanswerable questions. The dataset is divided into training, validation, and test sets, containing a total of 477 articles: 398 in the training set, 44 in the validation set, and 35 in the test set.  The following table summarizes the dataset statistics:

| Dataset    | Articles | Paragraphs | Questions | Mean of paragraphs in each article | Mean of questions in each paragraph |
|------------|----------|------------|-----------|-----------------------------------|-------------------------------------|
| Training   | 398      | 17,187     | 116,409   | 43.18                             | 6.77                                |
| Validation | 44       | 1,848      | 13,910    | 42.00                             | 7.53                                |
| Test       | 35       | 1,204      | 11,873    | 34.40                             | 9.86                                |
<p align="center">
  Table 1. Dataset statistics.
</p>
To better understand the distribution of paragraphs and questions within the dataset, we use the training set as an example and display the data in a histogram:

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="/data_statistics.png" height="300">
</div>
<p align="center">
  Figure 1. Distribution of Paragraphs per Article and Questions per Paragraph in the Training Set.
</p>


The histogram illustrates that the majority of articles contain between 20 and 60 paragraphs, with each paragraph typically having between 1 and 10 questions.


### 3.2 Qualitative Evaluation of the BERT Model

#### 3.2.1 Selected Cases

We chose several examples from the dataset to evaluate the BERT model's performance:

**Case 1:**

*Context:* The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.

*Question:* Who designed the Eiffel Tower?

*Model's Answer:* Gustave Eiffel

*Correct Answer:* Gustave Eiffel

In this example, the BERT model successfully identifies the correct answer within the context.

**Case 2:**

*Context:* The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, generally built along an east-to-west line across the historical northern borders of China.

*Question:* What materials were used to build the Great Wall of China?

*Model's Answer:* stone, brick, tamped earth, wood, and other materials

*Correct Answer:* stone, brick, tamped earth, wood, and other materials

The BERT model successfully extracts the correct answer from the context, showing its ability to handle more complex questions.

#### 3.2.2 Failure Case Analysis

Despite its success in many cases, the BERT model still exhibits some limitations. Analyzing failure cases can help us understand these limitations and guide future improvements:

**Failure Case 1:**

*Context:* Sir Isaac Newton was an English mathematician, physicist, astronomer, theologian, and author who is widely recognized as one of the most influential scientists of all time, and a key figure in the scientific revolution.

*Question:* What was Isaac Newton's nationality?

*Model's Answer:* English mathematician

*Correct Answer:* English

In this example, the BERT model fails to extract the precise answer, instead providing extra information. This may be due to the model's difficulty in distinguishing between relevant and irrelevant details within the context.

**Failure Case 2:**

*Context:* The Battle of Gettysburg was fought July 1-3, 1863, in and around the town of Gettysburg, Pennsylvania, by Union and Confederate forces during the American Civil War.

*Question:* What year did the Battle of Gettysburg take place?

*Model's Answer:* July 1-3

*Correct Answer:* 1863

In this case, the model incorrectly focuses on the specific dates of the battle rather than the year. This highlights the model's occasional inability to grasp the precise scope of a question.

By analyzing both successful and failed cases, we can better understand the BERT model's strengths and weaknesses, informing future refinements and improvements to the model.

### 3.3 Ablation Studie

#### 3.3.1 Hyperparameter Tuning

In this experiment, we chose the following hyperparameter settings:

**Batch size**: We experimented with batch sizes of 8, 16, and 32. The results were as follows:

+ Batch size of 8: Exact accuracy: 68.82\%, F1 score: 75.41\%.
+ Batch size of 16: Exact accuracy: 69.74\%, F1 score: 76.53\%.
+ Batch size of 32: Exact accuracy: 68.86\%, F1 score: 75.64\%.

The model's performance was slightly higher when the batch size was set to 16, compared to 8 and 32. This indicates that in this experiment, a larger batch size (e.g., 32) did not result in further performance improvements. This may be because a larger batch size could lead to a reduced generalization ability of the model during training. It is worth noting that the optimal batch size value may vary depending on the dataset, task, and hardware conditions. Therefore, further testing and adjustments may be necessary in real-world scenarios.

**Learning rate (Adam)**:We experimented with learning rates of 5e-05, 3e-05, and 1e-05. The results were as follows:

+ Learning rate of 5e-05: Exact accuracy: 69.74\%, F1 score: 76.53\%.
+ Learning rate of 3e-05: Exact accuracy: 69.03\%, F1 score: 75.96\%.
+ Learning rate of 1e-05: Exact accuracy: 67.55\%, F1 score: 74.56\%.

A higher learning rate (5e-05) yielded the best performance in this experiment. This suggests that a larger learning rate can help the model converge to a better solution more quickly during training, resulting in better performance. However, it is important to note that an overly large learning rate can cause catastrophic forgetting during training, so the learning rate should be adjusted according to the specific task and dataset. In this experiment, a learning rate of 5e-05 showed the best results, but other tasks and datasets may require different learning rate settings.

**Max sequence length:** This parameter sets the maximum input sequence length, affecting the model's performance in reading comprehension tasks.

In our experiments, we tested max seq lengths of 384 and 512:

- 384: Exact accuracy: 68.82%, F1 score: 75.41%.
- 512: Exact accuracy: 67.35%, F1 score: 74.20%.

A length of 384 yielded slightly better performance, possibly due to efficient text processing and reduced noise. Optimal max seq length may vary for different datasets and tasks, requiring further experimentation and adjustments.

In addition, we employed the following optimization strategies:

**Optimizer:** We used a modified Adam optimizer with weight decay, which prevents overfitting and maintains model stability. Weight decay is an L2 regularization term added to the loss function, helping the model generalize better and avoid performance degradation.

**Warmup and learning rate decay:** Using warmup and linear decay, we increased the learning rate initially, preventing issues from large weight updates. After warmup, the learning rate decreases gradually, enabling finer adjustments and reducing performance fluctuations on the validation set.

#### 3.3.2Neural Network Architecture

In this assignment, we employed different pre-trained model architectures, including BERT, BERT-SQuAD2, and ALBERT, to improve the performance of the reading comprehension task. We tried different scales of BERT and ALBERT models, as well as models fine-tuned on the SQuAD2 task, to evaluate the performance of each model on the current task and select the most suitable model.

**BERT-SQuAD2:** A BERT model fine-tuned on SQuAD2 for reading comprehension tasks. It has better initial performance but using it is discouraged due to ethical concerns. Best result: exact: 81.25%, F1: 87.73%.

**ALBERT:** A lightweight version of BERT with reduced parameters and faster training. We tested Albert-base, Albert-large, Albert-xlarge, and Albert-xxlarge. Best result (Albert-large): exact: 81.43%, F1: 84.86%.

Experimenting with different BERT and ALBERT models helps us evaluate their performance and select the most suitable one for reading comprehension tasks, crucial for achieving optimal results.


## 4 Conclusion:

In summary, our experiments focused on adjusting hyperparameters, optimization strategies, and pre-trained model architectures for reading comprehension tasks. We found a balance between training speed and model stability with the right learning rate, batch size, and gradient accumulation steps. The optimizer and learning rate decay strategies improved the model's generalization and prevented overfitting.

BERT and ALBERT showed promising results, while BERT-SQuAD2, though achieving high scores, was discouraged for ethical reasons. The experiments provided insights into each model's strengths and weaknesses, guiding our selection for the task. These results highlight the importance of careful selection and experimentation in achieving optimal performance in NLP tasks.
