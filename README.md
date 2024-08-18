# Real-Time Emotion Recognition Using Facial Expressions

This repository contains the code for a real-time emotion recognition system developed using OpenCV and TensorFlow. The project aims to detect and classify human emotions from live video input captured via a webcam, leveraging both traditional computer vision techniques and deep learning models.

## Project Overview

This project focuses on creating a robust emotion detection system that can process live video input, detect faces, and predict emotional states using a deep learning model. The key components of this system include:
- **OpenCV**: Used for real-time face detection via Haar Cascades.
- **TensorFlow**: A pre-trained deep learning model classifies emotions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Methodology

The system integrates two major components:
1. **Face Detection**: Using Haar Cascade classifiers to detect faces from the live video feed.
2. **Emotion Classification**: Leveraging a convolutional neural network (CNN) built with TensorFlow to classify the detected facial expressions into emotional categories.

### Key Features:
- Real-time processing of live video input.
- Emotion classification with high accuracy using deep learning techniques.
- Modular and scalable code architecture, making it easy to extend or modify.

## Technologies Used
- **OpenCV**: For real-time face detection.
- **TensorFlow/Keras**: For deep learning and emotion classification.
- **Python**: The core programming language used for implementation.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/real-time-emotion-recognition.git

2. Install the required dependencies:

  pip install opencv-python-headless
  pip install tensorflow
  pip install numpy
  pip install matplotlib
  pip install seaborn
  pip install pandas

3. Run the emotion detection script:
  python emotion_detector.py

## Usage
The system captures video input from your webcam, detects faces, and then classifies emotions in real-time.
The deep learning model processes grayscale images of size 48x48 pixels and classifies them into one of the seven emotions.

## Model Details
# The model was trained on the Face Expression Recognition Dataset from Kaggle and consists of the following layers:

Convolutional layers for feature extraction.
Max Pooling layers to reduce dimensionality.
Dropout layers to prevent overfitting.
Dense layers for classification.
The model achieved an accuracy of 77.8% on the test set, with particularly strong performance in detecting emotions such as 'Happiness' and 'Surprise.'

## Results
Training Accuracy: 77.8%
Validation Accuracy: Varies, with some fluctuations due to potential overfitting.
Precision and Recall: Strongest for emotions like 'Happiness' and 'Surprise', while emotions like 'Disgust' and 'Fear' showed lower detection rates.


## Future Work
# Further improvements could be made by:
Training with a more diverse dataset to improve accuracy across all emotions.
Experimenting with more advanced neural network architectures.
Extending the model to handle varying cultural expressions and lighting conditions.

