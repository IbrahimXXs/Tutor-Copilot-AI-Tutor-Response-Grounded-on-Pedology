# Tutor Copilot: AI Tutor Response Grounded on Pedology

**Authors:**  
- **Besher Hassan**, MBZUAI ([Besher.Hassan@mbzuai.ac.ae](mailto:Besher.Hassan@mbzuai.ac.ae))  
- **Ibrahim Alsarraj**, MBZUAI ([Ibrahim.Alsarraj@mbzuai.ac.ae](mailto:Ibrahim.Alsarraj@mbzuai.ac.ae))  
- **Chenxi Wang**, MBZUAI ([Chenxi.Wang@mbzuai.ac.ae](mailto:Chenxi.Wang@mbzuai.ac.ae))

---

## Project Overview

Tutor Copilot is a system designed to detect and recommend effective pedagogical strategies in tutor-student dialogues using cutting-edge Natural Language Processing (NLP) techniques. It incorporates two classification layers:
1. **Binary Classification:** Identifies whether a pedagogical strategy is present in a given dialogue.
2. **Fine-Grained Classification:** Determines the specific pedagogical strategy used from 8 predefined categories.

In addition, the system recommends appropriate strategies based on the dialogue context, leveraging state-of-the-art models like BERT and GPT-4o for data augmentation.

---

## Repository Structure

### **Datasets**
- **`bridge_data_task1.csv`**: Binary classification dataset for detecting the presence of pedagogical strategies.
- **`bridge_data_task2_and_task3.csv`**: Dataset for fine-grained classification, containing expert tutor responses and corresponding strategy labels.
- **Augmented Datasets:**
  - **`deduplicated_balanced_shuffled_Train_dataset_GPT4o.csv`**: Augmented fine-grained classification data.
  - **`bridge_train_Augmented_data.csv`**: Augmented binary classification data.
  - **`exp_data.csv`**: Combined predictions for majority voting methods.
  - **`exp_data_prob.csv`**: Probability outputs for probabilistic voting methods.

### **Code Files**
1. **Binary Classification:**
   - **`Part1_Binary_Pedagogical_Strategy_Detection_SMOTE.ipynb`**: Implements traditional models with SMOTE balancing.
   - **`Part2_Binary_Pedagogical_Strategy_Detection_DataAugmentation.ipynb`**: Fine-tunes BERT on GPT-augmented binary data.
2. **Fine-Grained Classification:**
   - **`Part1_Multi_Class.ipynb`**: Fine-tunes BERT for multi-class classification.
   - **`Part2_GPT_Manual.ipynb`**: Manual augmentation process using GPT-4o.
   - **`Part3_Sklearn_Models.ipynb`**: Compares sklearn models for fine-grained tasks.
3. **Combination Methods:**
   - **`combination.py`**: Implements majority voting for combined predictions.
   - **`probability_combination.py`**: Implements probabilistic voting for ranked strategy recommendations.

---

## How to Use

### Prerequisites
- **Python 3.8 or higher**
- Install required dependencies:
  ```bash
  pip install pandas scikit-learn transformers sentence-transformers rank-bm25 matplotlib torch
