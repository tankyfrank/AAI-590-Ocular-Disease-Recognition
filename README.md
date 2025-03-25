# AAI-590-Ocular-Disease-Recognition

This repository contains the final code base for our AAI-590 Capstone Project, which focuses on developing a deep learning model for ocular disease recognition using fundus images and patient metadata. The goal of this project is to create a robust machine learning pipeline that enables automated classification of ocular diseases, aiding in early diagnosis and clinical decision-making.

## Project Overview
Course: AAI-590

Institution: University of San Diego

## Collaborators:

Sarah Durrani

Franklin Guzman

Hani Jandali

Instructor: Roozbeh Sadeghian

GitHub Repository URL: https://github.com/tankyfrank/AAI-590-Ocular-Disease-Recognition/edit/main/README.md

## Files and Their Purpose
File/Directory	Description
EDA.ipynb	Contains Exploratory Data Analysis (EDA), feature distribution, correlation analysis, and data preprocessing steps.
Feature_Engineering.ipynb	Code for feature selection, engineering, and transformations applied before model training.
cnn_v1.ipynb, cnn_v2.ipynb, cnn_v3.ipynb	Different versions of our CNN-based deep learning models used for disease classification.
ResnetMultiModal.ipynb	Implementation of ResNet-based multimodal model incorporating both image and metadata.
processed_ocular_disease.csv	Preprocessed dataset used for model training and validation.
ocular-disease-recognition.zip	Raw dataset containing fundus images and patient metadata.
README.md	This document, explaining the project structure and workflow.
Project Workflow
The project follows a structured pipeline to ensure reproducibility and effective model evaluation:

1. Data Collection & Cleaning

  Importing fundus images and metadata from ODIR-5K dataset

  Handling missing values, encoding categorical variables, and standardizing numerical features

2️. Exploratory Data Analysis (EDA)

  Visualizing feature distributions, correlation heatmaps, and class imbalances

  Identifying outliers and patterns in patient data

3️. Model Design & Building

  Implementing CNN architectures (Basic CNN, ResNet)

  Multimodal Learning (combining fundus images + patient metadata)

4️. Model Training & Optimization

  Training models with augmented datasets using SMOTE for class imbalance handling

  Hyperparameter tuning using GridSearchCV

5️. Model Evaluation & Analysis

  Computing classification metrics (accuracy, precision, recall, F1-score)

  Generating confusion matrices and ROC curves

## Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your_username/AAI-590-Ocular-Disease-Recognition.git
cd AAI-590-Ocular-Disease-Recognition
2. Install Required Dependencies
bash
Copy
Edit
pip install -r requirements.txt
(Ensure Python 3.8+ is installed before running the above command.)

3. Run Jupyter Notebooks
bash
Copy
Edit
jupyter notebook
Open any .ipynb file inside the /notebooks/ directory and run cells as needed.

## Key Technologies Used
Programming Language: Python 3.8+

Deep Learning Frameworks: TensorFlow, Keras, PyTorch

Data Processing: Pandas, NumPy, OpenCV

Visualization Tools: Matplotlib, Seaborn

Machine Learning Models: Convolutional Neural Networks (CNN), ResNet

Collaboration and Version Control
Git Workflow
Branching: Each team member works on a separate branch (feature-eda, model-training, hyperparameter-tuning, etc.).

Commits: Meaningful commit messages documenting changes.

Merging: Regular integration to maintain an organized codebase.

GitHub Issues: Used for tracking progress and resolving challenges.

## Future Improvements
Deploying the model via Flask API for real-world usage.

Experimenting with Transformer-based models (ViTs) for medical imaging.

Implementing Explainable AI (XAI) techniques to provide model interpretability.

## Acknowledgments
This project was completed as part of AAI-590 at the University of San Diego. Special thanks to our instructor and teammates for their collaboration and feedback.

## License
This project is for educational purposes only. If you use or modify this work, please provide appropriate attribution.
