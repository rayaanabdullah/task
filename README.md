# Project Overview

## Toxic Content Classification Model

This project aims to develop a machine learning model to detect and classify toxic content in online feedback. The primary goal is to predict whether a comment contains toxic content, using a variety of labels such as **abusive**, **vulgar**, **menace**, **offense**, and **bigotry**. The final prediction, however, focuses only on the **toxic** label, which indicates the presence of toxic language in a comment.

### Key Objectives:
- Classify feedback into different categories of toxicity.
- Predict whether a piece of feedback is toxic (binary classification).
- Address challenges such as language diversity and class imbalance in the dataset.

## Dataset Description

The dataset is divided into two main files:

1. **train.csv**: Contains labeled training data with the following columns:
   - **id**: Unique identifier for each comment.
   - **feedback_text**: The text of the feedback.
   - **toxic**: Binary label (1 for toxic, 0 for not toxic).
   - **abusive**: Binary label for abusive content.
   - **vulgar**: Binary label for vulgar language.
   - **menace**: Binary label for threatening language.
   - **offense**: Binary label for insults.
   - **bigotry**: Binary label for identity-based hate.

2. **test.csv**: Contains unlabeled data for evaluation during the testing phase.

### Language Distribution:
While the training data is in English, the testing and validation phases involve multiple languages. This introduces the need for handling multilingual content effectively.

### Class Imbalance:
There is a class imbalance in the dataset where toxic comments might not be evenly distributed across the different categories. The primary evaluation metric will focus on identifying whether the comment is **toxic** or not, though granular labels (such as abusive, vulgar, etc.) will also be useful during training.

## Model Implementation

### Steps to Build the Model:
1. **Exploratory Data Analysis (EDA)**:
   - Visualize label distributions.
   - Analyze text features like sentence length, common words, etc.
   - Handle missing values or outliers.

2. **Text Preprocessing**:
   - Tokenize the feedback text.
   - Convert text to lowercase and remove special characters.
   - Apply techniques like stemming or lemmatization.
   - Use feature extraction methods like TF-IDF or Transformer embeddings.

3. **Model Implementation**:
🧠 Model Implementation Details

🌲 **Random Forest**
The Random Forest model was optimized using RandomizedSearchCV with a 5-fold cross-validation approach. A wide range of hyperparameters were explored, including the number of estimators, max depth, minimum samples per split and leaf, class weighting, and maximum feature selection methods. TF-IDF features were used for training the model, and class imbalance was handled using resampling techniques. The best estimator obtained from the random search was used for final training and evaluation.

🔁 **LSTM (Long Short-Term Memory)**
The LSTM model was implemented using PyTorch, and input data was tokenized using the BERT tokenizer. A custom Dataset class was created to handle text preprocessing. The model architecture included an embedding layer, a two-layer LSTM, dropout regularization, and a fully connected output layer. The model was trained using cross-entropy loss and the Adam optimizer over 5 epochs. Performance metrics including accuracy and ROC-AUC were logged per epoch, and class imbalance was a significant challenge.

🌐 **XLM-RoBERTa**
XLM-RoBERTa was fine-tuned using the HuggingFace `Trainer` API for sequence classification. The pre-trained 'xlm-roberta-base' model was used, with a binary classification head. Training arguments included early stopping, learning rate scheduling, and evaluation on the validation set per epoch. This approach is particularly suited for multilingual tasks, offering robustness in handling language diversity. Training was performed on a GPU when available to speed up the process.

**Why these three model choosen for Implementation**

🔹 Random Forest
Random Forest is a powerful baseline model that performs well on structured data and provides interpretable results. It can handle class imbalance through techniques like class weighting and bootstrapping. Given your dataset’s multi-label binary format and the need for overall “toxic” detection, Random Forest can effectively learn from engineered features like TF-IDF. It’s also relatively fast to train and evaluate, making it suitable for benchmarking and quick iterations before moving to more complex models.

🔹 LSTM (Long Short-Term Memory)
LSTM networks are ideal for processing sequential data, such as text, where context and word order matter. They can capture long-term dependencies and nuances in sentence structure—crucial for distinguishing subtle toxic language patterns. This is particularly useful in multi-label classification tasks involving diverse linguistic expressions. LSTM models can generalize well to multilingual content by learning patterns from longer feedback texts, which helps address the multilingual challenge in your validation/test sets.

🔹 XLM (Cross-lingual Language Model)
XLM is specifically designed for multilingual NLP tasks. It leverages Transformer architecture and cross-lingual embeddings, making it exceptionally suitable for your dataset, which contains English during training but expects predictions on various languages. Fine-tuning XLM allows the model to generalize across languages and handle code-switching or non-English content more robustly. Its contextual understanding significantly boosts performance in identifying toxic content across cultures and linguistic variations.

5. **Model Evaluation**:
   - Compute metrics like accuracy, precision, recall, and F1-score.
   - Visualize performance with Confusion Matrices and AUC-ROC curves.

6. **Model Tuning**:
   - Fine-tune hyperparameters using Random Search.

### Deliverables:
- Two Jupyter notebooks or Python files implementing the models.
- Performance analysis reports, including metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
- Full explanation of the choices made in terms of model architecture and evaluation.

## How to Run the Code

1. Clone this repository.
2. Install the required dependencies from `requirements.txt`.
3. Run the notebooks or Python files for both models:
   - `model1_implementation.ipynb` (Random Forest and LSTM models).
   - `model2_implementation.ipynb` (XLM model).

4. Evaluate the models using the test dataset and review the performance metrics.

## Model Evaluation Results

- The model's performance will be evaluated based on several key metrics such as:
  - **Accuracy**: Overall prediction accuracy.
  - **Precision**: Precision of predicting toxic comments.
  - **Recall**: Sensitivity of the model in detecting toxic content.
  - **F1-Score**: Harmonic mean of precision and recall.
  - **AUC-ROC**: Area under the ROC curve to assess classification performance.

## Additional Notes:
- This project is designed to simulate real-world content moderation challenges.
- Handling multilingual content and class imbalance are key aspects of the solution.
