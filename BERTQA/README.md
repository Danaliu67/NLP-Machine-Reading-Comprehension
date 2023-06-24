# BERT Model Assignment

This repository contains the code for the BERT model assignment. The task requires the use of a more advanced model and the adoption of practical techniques in hyperparameter tuning to improve model performance. **All experiments were conducted on an offline server with no internet connection. All required data and models were downloaded beforehand.**

**Note: This project requires the installation of the Hugging Face Transformers library. You can find the GitHub repository [here](https://github.com/huggingface/transformers). The pre-trained weights have been downloaded locally, so the code does not reflect this.**

## Model Descriptions

- BERT (Bidirectional Encoder Representations from Transformers): BERT is a pre-trained model based on the Transformer architecture that captures bidirectional context relationships in text. This assignment includes both bert-base and bert-large models.

- BERT-SQuAD2: The BERT model fine-tuned on the SQuAD2 task, specifically designed for reading comprehension tasks.

- ALBERT (A Lite BERT): ALBERT is a lightweight version of the BERT model, which reduces the number of model parameters and increases training speed. This assignment includes Albert-base, Albert-large, Albert-xlarge, and Albert-xxlarge models.

In this assignment, we used different scales of BERT and ALBERT models, as well as models fine-tuned on the SQuAD2 task, to improve model performance. By training and testing on various model scales, we can evaluate the performance of each model and choose the most suitable one for the current task.

## Modified and Added Files

1. `main.py`: Main training script
2. `utils_squad.py`: Utility functions for SQuAD data processing
3. `modeling_bert.py`: BERT model implementation
4. `utils_squad_evaluate.py`: Utility functions for SQuAD evaluation
5. `eval-albert-L.py`: Evaluation script for the Albert-large model
6. `eval-albert.py`: Evaluation script for the Albert-base model
7. `eval-B.py`: Evaluation script for the BERT-base model
8. `main-albert-L.py`: Training script for the Albert-large model
9. `main-albert-xL.py`: Training script for the Albert-xlarge model
10. `main-albert-xxL.py`: Training script for the Albert-xxlarge model
11. `main-albert.py`: Training script for the Albert-base model
12. `main-L.py`: Training script for the BERT-large model
13. `main-L-sq.py`: Training script for the BERT-large-SQuAD2 model
14. `main-B.py`: Training script for the BERT-base model
15. `main-B-sq.py`: Training script for the BERT-base-SQuAD2 model

Note: The Albert-xlarge and Albert-xxlarge models were not trained due to their large size.

## Chosen Model

The Albert-large model has been chosen for the final submission due to its superior performance compared to the other models. The results for the Albert-large model are:

- Exact: 81.42845110755496
- F1: 84.86029930097264
- 0.5 * EM + 0.5 * F1: 83.1443752042638

## Additional Files

- `log/pytorch_model.bin`: The best-performing model (Albert-large) as a PyTorch binary file.
- `log`: This folder contains log files for different models and training runs. Each `output-model` folder contains a backup of the config (training parameters) and results (trainingoutputs) for each training session.