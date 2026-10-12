 SMS Spam Classifier 📩

About the Project
This project is a machine learning-based SMS Spam Classifier developed using Python and Scikit-learn. It classifies SMS messages into two categories: **Spam** (unwanted messages) and **Ham** (legitimate messages).

The model uses **TF-IDF Vectorization** to convert text messages into numerical features and **Logistic Regression** to predict whether a message is spam or ham.

Features
- Loads and processes an SMS dataset using Pandas.
- Converts text messages into numerical features using TF-IDF.
- Splits the dataset into training and testing sets.
- Trains a Logistic Regression classification model.
- Evaluates performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Predicts whether a user-entered message is Spam or Ham.

Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

Model Evaluation
The model is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix to assess its classification performance.

Learning Outcomes
- Text preprocessing and feature extraction
- TF-IDF vectorization
- Logistic Regression for binary classification
- Training and testing machine learning models
- Evaluating classification performance
- Making predictions on new text data

Future Improvements
- Compare different machine learning algorithms.
- Experiment with TF-IDF parameters.
- Tune the classification threshold using validation data.
- Build a simple web interface for real-time message classification.

Dataset and Attribution

This project uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), originally published by Tiago Almeida and José Hidalgo through the UCI Machine Learning Repository.

Original source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0).

The dataset is used for educational and machine learning purposes. Credit is given to the original dataset creators.

This project implements an SMS spam classification model using TF-IDF vectorization and Logistic Regression.

