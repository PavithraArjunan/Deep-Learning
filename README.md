This project focuses on classifying breast cancer as benign or malignant using a simple Neural Network (NN). The model is trained on the Breast Cancer dataset available in `sklearn.datasets` and utilizes deep learning techniques for binary classification.

## Dependencies
The following libraries are required to run this project:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from tensorflow import keras
```

## Dataset
The dataset used is the Breast Cancer dataset from `sklearn.datasets`. It contains features extracted from digitized images of breast mass.

## Steps Involved

### 1. Load the Dataset
```python
data = sklearn.datasets.load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
```

### 2. Data Preprocessing
- Checking for missing values
- Splitting dataset into training and testing sets
```python
print(df.isnull().sum())
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. Building the Neural Network
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),  # Input layer
    keras.layers.Dense(20, activation='relu'),  # Hidden layer
    keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])
```

### 4. Compiling the Neural Network
```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Correct loss function for binary classification
              metrics=['accuracy'])
```

### 5. Training the Model
```python
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
```

### 6. Evaluation & Visualization
```python
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
```

## Results
- The model is evaluated using accuracy and loss metrics.
- Final accuracy and loss plots are visualized.

## Conclusion
This project demonstrates how a simple neural network can be used for breast cancer classification using structured data. The model can be further improved with hyperparameter tuning and advanced architectures.

