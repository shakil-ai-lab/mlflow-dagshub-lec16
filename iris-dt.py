import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import mlflow
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

max_depth = 3
n_estimators = 30

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a Random Forest Classifier

# Start an MLflow run
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("test_experiment")
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(model, "model")
    mlflow.set_tag("author", "shakil")
    mlflow.set_tag("model_type", "RandomForestClassifier")
    print(f"Accuracy: {accuracy}")