# Waste-Classifier
# Waste Classifier

A deep learning project for classifying waste images into multiple categories using a Convolutional Neural Network (CNN) based on DenseNet169 pretrained on ImageNet.

## Overview

This project aims to build an image classification model that can identify different types of waste such as:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The model is trained using a Kaggle dataset and implemented in Python with TensorFlow/Keras.

## Project Goals

- Download and preprocess the waste image dataset
- Analyze class distribution and image properties
- Split the data into train, validation, and test sets
- Train a transfer learning model
- Evaluate model performance
- Save the trained model
- Use the trained model for prediction on new waste images

## Dataset

The project uses the Trash Type Image Dataset from Kaggle.

Dataset link:
https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset

The dataset contains images divided into classes:
- cardboard
- glass
- metal
- paper
- plastic
- trash

## Model Architecture

The project uses a transfer learning approach with:

- DenseNet169 as the base model
- ImageNet pretrained weights
- GlobalAveragePooling2D
- Dropout
- Softmax output layer for 6 classes

## Project Structure

```text
CNN/
├── model.ipynb              # Notebook containing the full training and evaluation workflow
├── Trash.jpeg               # Example input image
├── Trash2.jpeg              # Example input image
├── TrashType_Image_Dataset/ # Downloaded dataset
├── waste-classifier/        # Project folder or exported version
├── README.md                # Project documentation
└── waste_classifier_densenet201.keras   # Saved trained model
```

## Requirements

Make sure the following Python packages are installed:

```bash
pip install tensorflow keras pandas numpy matplotlib seaborn scikit-learn pillow kagglehub
```

## Setup

1. Clone or download the project
2. Open the notebook in Jupyter or VS Code
3. Run the cells in order
4. Ensure Kaggle access is configured for dataset download

For KaggleHub:
```python
!pip install -q kagglehub
```

## Running the Notebook

Open:
```bash
model.ipynb
```

Then run the cells sequentially:
- dataset download
- EDA and visualization
- train/validation/test split
- preprocessing
- model building
- training
- evaluation
- prediction

## Training

The notebook builds a DenseNet169 model and trains it with:

- Adam optimizer
- Sparse categorical cross-entropy loss
- Early stopping
- Learning rate reduction on plateau
- Class-weight balancing

## Evaluation

The model is evaluated using:
- validation accuracy
- validation loss
- confusion matrix
- classification report
- test accuracy and test loss

## Model Saving

The trained model is saved as:

```python
model.save("waste_classifier_densenet201.keras")
```

## Prediction Example

A sample prediction is done by loading an image, resizing it to 224x224, normalizing it, and passing it into the trained model:

```python
from PIL import Image
import numpy as np

class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

image = Image.open("Trash2.jpeg").convert("RGB")
image = image.resize((224, 224))
image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

pred = model.predict(image_array, verbose=0)
predicted_class = class_names[np.argmax(pred)]
confidence = pred[0][np.argmax(pred)] * 100

print("Prediction:", predicted_class)
print("Confidence:", round(confidence, 2), "%")
```

## Notes

- The project uses transfer learning to reduce training time and improve accuracy.
- Training may take time depending on your GPU/CPU and dataset size.
- If you want better performance, you can try:
  - more epochs
  - tuning learning rate
  - using a larger pretrained backbone
  - using image augmentation more aggressively

## License

This project is for educational and research purposes.

## Author

Your Name / GitHub Username

## GitHub Repository

You can push this project to GitHub with:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repository>.git
git push -u origin main
```

## Future Improvements

- Add a web interface using Streamlit or Flask
- Create a mobile-friendly deployment
- Add more categories
- Improve model accuracy with additional tuning
- Export model for production use
