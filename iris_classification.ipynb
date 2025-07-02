# Iris Flower Classification Project

# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Load Dataset ---
iris = sns.load_dataset('iris')
print(iris.head())

# --- Encode Labels ---
le = LabelEncoder()
iris['species'] = le.fit_transform(iris['species'])  # setosa: 0, versicolor: 1, virginica: 2

# --- Feature Selection ---
X = iris.drop('species', axis=1)
y = iris['species']

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Neural Network Model ---
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, activation='relu', solver='adam', random_state=42)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred = model.predict(X_test)

# --- Evaluation ---
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# --- Confusion Matrix ---
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


