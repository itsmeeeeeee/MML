# README: Deep Learning and Machine Learning Approaches Including Ensemble Techniques for Multimodal Sentiment Analysis

## Project Title
**Deep Learning and Machine Learning Approaches Including Ensemble Techniques for Multimodal Sentiment Analysis: Memotion Classification**

## Project Description
This project focuses on multimodal sentiment analysis using the Memotion dataset, which includes image-based memes with labeled sentiment data. Originally, the dataset included 6,992 entries categorized into positive (4,160), neutral (2,201), and negative (631) sentiments. For the purpose of binary classification, neutral labels were excluded, resulting in a dataset of 4,791 entries, distinguishing memes as negative (labeled as 0) and positive (labeled as 1). Three primary approaches—MMFA (Multimodal Fusion with Self Attention), MMF (Multimodal Fusion without self-attention), and Sklearn Ensemble—are explored. Each utilizes five classifiers and employs an ensemble method based on majority voting for robust sentiment classification.

## Dataset
- **Primary Source**: Memotion dataset from [Kaggle: Memotion Dataset 7k](https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k/data).
- **Data Storage and Features**:
  - Image folders and labeled data are utilized.
  - Extracted features include image and text embeddings, hosted on Google Drive:
    - [Colab Data Folder](https://drive.google.com/drive/folders/1nNVU4FFF0R4Z2tA4d4b8EjAyvRMLByWs?usp=drive_link)
    - [Extracted Features](https://drive.google.com/drive/folders/1TfjAZEWsVyoQQMX_TIl-TXeuf3rU3VAR?usp=sharing)

## Feature Extraction
- **File**: `Features_extraction.ipynb`
- **Description**: This notebook details the process of extracting image and text features using pre-trained models, specifically BERT for text and ResNet-50 for images. 

## Implementation and Model Description

- **MMFA (Multimodal Fusion with Self-Attention)**: This model incorporates self-attention mechanisms to better integrate and analyze the features extracted from both image and text data. Following the individual classifier predictions, an ensemble method using majority voting is applied to combine the results for enhanced accuracy.
- **MMF (Multimodal Fusion without Self-Attention)**: Operates similarly to MMFA but removes the self-attention component, focusing on direct fusion of features. This approach also concludes with an ensemble method using majority voting.
- **Sklearn Ensemble**: Utilizes a traditional machine learning approach with five common algorithms: Decision Tree, Multilayer Perceptron, Logistic Regression, Adaptive Boosting, and Support Vector Machine. Each model's results are then aggregated using an ensemble strategy (majority voting) to provide comparative insights against the deep learning methods.
  
All models are executed within Google Colab, ensuring easy access and reproducibility.

## Installation and Usage
No installation is required as the project is developed in Google Colab. Users can access and execute all files directly in the Colab environment.

## Results
This table summarizes the performance of models from the Memotion Competition, evaluated based on their macro-average F1 scores, with further details available in the [Memotion Competition Analysis](https://arxiv.org/pdf/2008.03781).


| Model             | Macro F1-score | Comparison with baseline (+/-) |
|-------------------|----------------|--------------------------------|
| Vkeswani IITK     | 0.35466        | (+)0.13701                     |
| Guoym             | 0.35197        | (+)0.13432                     |
| Aihaihara         | 0.35017        | (+)0.13252                     |
| Sourya Diptadas   | 0.34885        | (+)0.13120                     |
| Irina Bejan       | 0.34755        | (+)0.12990                     |
| **SemEval-Baseline** | 0.2176      | -                              |

### Our Approaches and Results

#### Deep-Learning Approach 1: Multimodal Fusion with Self-Attention (MMFA)
- **Results**:
  | Model              | Macro F1-score | Accuracy |
  |--------------------|----------------|----------|
  | CNN                | 0.4730         | 0.8975   |
  | LSTM               | 0.4710         | 0.8906   |
  | RNN                | 0.4720         | 0.8940   |
  | FFNN with Softmax  | 0.4734         | 0.8993   |
  | MLP                | 0.5056         | 0.8993   |
  | Ensemble, Mj. Voting | 0.4701       | 0.8873   |

#### Deep-Learning Approach 2: Multimodal Fusion (MMF)
- **Results**:
  | Model              | Macro F1-score | Accuracy |
  |--------------------|----------------|----------|
  | MLP                | 0.4735         | 0.8958   |
  | CNN                | 0.4879         | 0.8645   |
  | RNN                | 0.4828         | 0.8819   |
  | FFNN with Softmax  | 0.4730         | 0.8975   |
  | LSTM               | 0.5008         | 0.8906   |
  | Ensemble, Mj. Voting | 0.4701       | 0.8873   |

#### Machine Learning Approach Using Scikit-Learn
- **Results**:
  | Model              | Macro F1-score | Accuracy |
  |--------------------|----------------|----------|
  | Decision Tree      | 0.4892         | 0.7445   |
  | Multilayer Perceptron | 0.5014      | 0.8112   |
  | Logistic Regression| 0.4964         | 0.8477   |
  | Adaptive Boosting  | 0.4971         | 0.8728   |
  | Support Vector Machine | 0.4701     | 0.8873   |
  | Ensemble, Mj. Voting | 0.4740       | 0.8727   |

## Conclusion and Observations
Throughout the project, we observed that the models required a large dataset to perform optimally. Due to the size of the available dataset (approximately 4,791 entries after filtering out neutral sentiments), the ensemble approach, which ideally benefits from larger datasets, did not perform as expected. The approach itself is sound; however, the limited data volume may have constrained the effectiveness of the ensemble methods. For validation, we employed a 5-fold cross-validation technique to ensure the robustness and generalizability of our models across different subsets of data.

### Research Questions and Answers

**How effective are the developed Deep Learning and Machine Learning models compared to the baseline and the top participants of the Memotion competition in Task A of sentiment analysis?**

Our models significantly outperformed the baseline, which had a Macro-F1 score of 0.21765. The best model, MLP in the MMFA approach, achieved a Macro F1-score of 0.5056, demonstrating superior performance. Additionally, when compared to the top participants whose scores ranged from 0.34600 to 0.35466, our models demonstrated competitive and even superior results, with several models exceeding these scores substantially.

This indicates that our approaches are not only effective in surpassing the baseline but also competitive within the context of the Memotion competition, offering robust alternatives to traditional sentiment analysis methods.
