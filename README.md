
# Text Classification Using CNN and LSTM

## Overview
This project demonstrates the use of **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks for text classification, specifically addressing the sentiment analysis of article reviews. It includes steps for data exploration, preprocessing, and model implementation using neural networks. 

---

## Table of Contents
- [Introduction](#introduction)
- [What is Text Classification?](#what-is-text-classification)
- [Applications of Text Classification](#applications-of-text-classification)
- [Models Used](#models-used)
- [Why CNN and LSTM?](#why-cnn-and-lstm)
- [Dataset and Problem Statement](#dataset-and-problem-statement)
- [Implementation](#implementation)
  - [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
  - [Word Embeddings](#word-embeddings)
  - [Model Building](#model-building)
- [Conclusion](#conclusion)
- [References](#references)

---

## Introduction
Unstructured data from emails, chats, webpages, and social media pose challenges for extracting actionable insights. Text classification, leveraging Natural Language Processing (NLP), automates the structuring of text data efficiently, enabling scalable data analysis.

---

## What is Text Classification?
Text classification, also known as text tagging or categorization, involves automatically assigning predefined categories or tags to unstructured text using NLP techniques.

---

## Applications of Text Classification
- **Sentiment Analysis:** Determines if text conveys positive or negative sentiment (e.g., brand monitoring).
- **Topic Detection:** Identifies the main topic of text (e.g., customer feedback analysis).
- **Language Detection:** Determines the language of incoming text (e.g., customer support tickets).

---

## Models Used
- **Convolutional Neural Network (CNN)**: Ideal for detecting specific patterns in text, useful for identifying expressions regardless of position.
- **Long Short-Term Memory (LSTM)**: A specialized recurrent neural network designed to retain context over sequences, mitigating the vanishing gradient problem often seen in standard RNNs.

---

## Why CNN and LSTM?
- CNNs effectively capture local patterns (expressions) in text but lack contextual awareness.
- LSTMs provide context by considering sequential order, beneficial for interpreting sentence meaning.
- Combining CNN and LSTM leverages strengths from both models, improving text classification accuracy.

---

## Dataset and Problem Statement
- **Dataset:** Article Review Dataset containing 999 highly polarized reviews (positive/negative).
  - Training: 699 text reviews
  - Testing: 299 text reviews

- **Objective:** Classify each review into positive or negative sentiment accurately.

---

## Implementation

### Data Exploration and Preprocessing
- Text cleaning, tokenization, removal of stop words.
- Visualization and understanding of the dataset characteristics.

### Word Embeddings
- Utilizing word embeddings to represent words as vectors, capturing semantic and syntactic word relationships.

### Model Building
Implemented text classification using neural networks in Keras:

#### CNN-based Text Classification
- Detect patterns through convolutional layers and apply pooling for feature extraction.

#### LSTM-based Text Classification
- Capture contextual dependencies through recurrent LSTM layers, beneficial for sequence data.

#### CNN + LSTM Combined Approach
- Integrating CNN for pattern recognition and LSTM for context handling to achieve superior text classification performance.

---

## Conclusion
The combined CNN-LSTM model achieves robust performance in sentiment classification tasks, efficiently capturing both local patterns and broader contextual information in textual data.

The purpose of the Text Classification Project using CNN and LSTM is to efficiently automate the categorization of textual data, specifically to classify article reviews into positive or negative sentiments. This automation facilitates rapid insight generation from unstructured data, reducing manual analysis efforts and providing scalable, cost-effective text processing.

#### How This Project Benefits a Company:
- Enhanced Decision-Making: Quickly identify customer sentiment, enabling responsive decision-making.

- Operational Efficiency: Automates manual processes, saving resources and time spent on analyzing textual feedback.

- Improved Customer Experience: Rapidly understand and address customer sentiment, leading to improved customer satisfaction.

#### Results Achieved:
- Developed an accurate, reliable text-classification model (CNN combined with LSTM) capable of effectively classifying reviews with high accuracy.

- Enabled real-time or near real-time insights into customer sentiment, allowing the company to swiftly address feedback and adjust strategies.

#### Impact on Company Improvement:
- Cost Reduction: Lowered costs associated with manual text analysis.

- Strategic Advantage: Gained faster insights into customer preferences, sentiment, and potential issues, enabling proactive rather than reactive strategies.

- Scalability: Enhanced capability to analyze large volumes of text data efficiently, supporting future growth and data-driven initiatives.
---

## References
- [Keras Documentation](https://keras.io/)
- [CNN & LSTM for Text Classification Tutorial](https://medium.com)

---



## License
This repository is available under the MIT License.
