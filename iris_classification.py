import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Streamlit app title
st.title("Iris Flower Classification")

# Sidebar for user input
st.sidebar.header("Model Parameters")
hidden_layers = st.sidebar.slider("Number of neurons in each hidden layer", 5, 50, 10)
max_iterations = st.sidebar.slider("Maximum iterations", 100, 2000, 1000, step=100)
test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, step=0.05)

# File uploader for custom dataset
uploaded_file = st.sidebar.file_uploader("Upload your Iris dataset (CSV)", type=["csv"])

# Load dataset
if uploaded_file is not None:
    iris = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset Preview:")
    st.write(iris.head())
else:
    iris = sns.load_dataset('iris')
    st.write("Default Iris Dataset Preview:")
    st.write(iris.head())

# Encode labels
le = LabelEncoder()
iris['species'] = le.fit_transform(iris['species'])  # setosa: 0, versicolor: 1, virginica: 2

# Feature selection
X = iris.drop('species', axis=1)
y = iris['species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train model
model = MLPClassifier(hidden_layer_sizes=(hidden_layers, hidden_layers), max_iter=max_iterations, activation='relu', solver='adam', random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {acc:.2f}")

# Classification report
st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
st.write("**Confusion Matrix:**")
fig, ax = plt.subplots()
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# Optional: Allow users to input custom data for prediction
st.header("Predict Iris Species")
st.write("Enter feature values for prediction:")
col1, col2, col3, col4 = st.columns(4)
sepal_length = col1.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = col2.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.5)
petal_length = col3.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.4)
petal_width = col4.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_species = le.inverse_transform(prediction)[0]
    st.write(f"**Predicted Species:** {predicted_species}")
