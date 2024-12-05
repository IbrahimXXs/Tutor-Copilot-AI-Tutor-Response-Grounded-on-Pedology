# README

This repository contains scripts and datasets for analyzing and predicting pedagogical strategies using various machine learning methods. Below is a detailed description of each file:

## File Descriptions

### 1. `combination.py`
- Implements a simple majority voting mechanism to combine the predictions of the following three algorithms:
  - **LPD** (Recommendation Based on Label Probability Distribution)
  - **BES** (BM25 and Embedding Similarity Combined Method)
  - **BERT-Augmented** (BERT model trained on augmented data)
- The script aggregates predictions from these three methods using majority voting to generate the final recommended strategy.
- Evaluates performance metrics such as accuracy, F1 score, recall, and precision.

### 2. `tradition_based.py`
- Implements a simple majority voting mechanism combining predictions from the following traditional machine learning models:
  - **SVM** (Support Vector Machine)
  - **Naive Bayes**
  - **Boosting**
- The script preprocesses data by removing stopwords, applying stemming, and converting text features into TF-IDF vectors.
- Aggregates predictions from these traditional models using majority voting to generate the final recommended strategy.
- Evaluates performance metrics for the combined traditional model.

### 3. `probability_combination.py`
- Implements a probabilistic voting mechanism to combine probability outputs from the following methods:
  - **BERT-Augmented** probabilities (weight: 0.5)
  - **LPD** probabilities (weight: 0.2)
  - **BES** probabilities (weight: 0.3)
- Calculates combined probabilities using weighted summation and generates a ranked list of recommended strategies instead of a single prediction.
- Outputs the recommendation probabilities for each strategy, sorted by likelihood.

### 4. `exp_data.csv`
- Provides baseline data for `combination.py`, containing:
  - Predictions from different models (LPD, BES, BERT-Augmented).
  - Basic dataset information, such as conversation history and pedagogical strategy labels.

### 5. `exp_data_prob.csv`
- Provides probability data for `probability_combination.py`, containing:
  - Probability predictions for all strategies from different models (LPD, BES, BERT-Augmented).
  - Used for implementing the probabilistic voting method.

## Usage Instructions

1. **Run Traditional Models:**
   Use the `tradition_based.py` script to combine SVM, Naive Bayes, and Boosting models through simple majority voting to predict pedagogical strategies.

2. **Run Enhanced Models:**
   Use the `combination.py` script to combine LPD, BES, and BERT-Augmented models through majority voting to generate the final recommendation.

3. **Run Probabilistic Combination:**
   Use the `probability_combination.py` script to combine probabilities from different models using weighted summation to generate ranked recommendations.

4. **Input Data:**
   - `exp_data.csv` is used by `combination.py` to provide prediction data.
   - `exp_data_prob.csv` is used by `probability_combination.py` to provide probability predictions.

5. **Evaluate Performance:**
   Each script outputs performance metrics such as accuracy, F1 score, recall, and precision for its respective methods.

## Requirements

- Python 3.8 or higher
- Required libraries:
  - `pandas`
  - `scikit-learn`
  - `xgboost`
  - `nltk`
  - `torch`
  - `transformers`
  - `rank_bm25`
  - `sentence-transformers`

## Notes

- `combination.py` and `probability_combination.py` use different voting mechanisms. Ensure the correct input files are used for each script.
- All models use the same label encoding mapping for consistency in results.