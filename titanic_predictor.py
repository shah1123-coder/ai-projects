# file structure

# 1. Import Libraries
# 2. Load Dataset
# 3. Explore Dataset
# 4. Clean/Preprocess
# 5. Visualize Patterns
# 6. Train/Test Split
# 7. Train Logistic Regression
# 8. Evaluate
# 9. Predict on Sample Input

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Data
def load_data(url):
    df = pd.read_csv(url)
    return df

# Step 2: Clean and Encode Data
def preprocess_data(df):
    df = df.drop(['Cabin', 'Name', 'Ticket'], axis=1)
    df = df.dropna()
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df

# Step 3: Train-Test Split (write this yourself!)
def train_test_split(X, y, test_size=0.2):
    """
    - Combine X and y together
    - Shuffle the rows randomly (use np.random.permutation)
    - Split into training and testing sets
    - Return: X_train, X_test, y_train, y_test
    """
    m = X.shape[0]
    indices = np.random.permutation(m)
    
    test_size = int(m * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    print(X_train.shape[0])
    print(X_test.shape[0])

    return X_train, X_test, y_train, y_test

# Step 6: Train Logistic Regression from Scratch
def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    """
    - Initialize weights and bias as zeros
    - For each epoch:
        - Compute linear output z = X.dot(weights) + bias
        - Apply sigmoid function: predictions = sigmoid(z)
        - Compute the error: predictions - y
        - Calculate gradients:
            - dw = (1/m) * X.T.dot(error)
            - db = (1/m) * sum(error)
        - Update weights and bias:
            - weights -= lr * dw
            - bias -= lr * db
    - Return the learned weights and bias
    """
    m = X.shape[0]
    n = X.shape[1]
    w = np.zeros(n)
    b = 0

    for i in range(epochs):
        z = X.dot(w) + b
        y_hat = 1 / (1 + np.exp(-z))
        error = y_hat - y
        
        dw = X.T.dot(error) * 1/m
        db = 1/m * np.sum(error) 

        w -= lr * dw
        b -= lr * db

    return w, b

    

# Step 7: Predict Function
def predict(X, weights, bias):
    """
    - Compute z = X.dot(weights) + bias
    - Apply sigmoid to get predicted probabilities
    - Convert probabilities to 0 or 1 using threshold (e.g., 0.5)
    - Return predictions as 0s and 1s
    """
    z = X.dot(weights) + bias
    y_hat = 1 / (1 + np.exp(-z))
    mask = y_hat >= 0.5
    y_hat[mask] = 1
    mask = y_hat < 0.5
    y_hat[mask] = 0
    return y_hat

# Step 8: Evaluate
def accuracy(y_true, y_pred):
    """
    - Compare y_true and y_pred element-wise
    - Count how many predictions are correct
    - Accuracy = correct / total
    """
    count = len(y_pred)
    correct = 0
    
    for i in range(count):
        if y_true.iloc[i] == y_pred.iloc[i]:
            correct += 1

    return correct / count

def main():
    df = load_data("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    data = preprocess_data(df)
    y = data["Survived"]
    X = data.drop("Survived", axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X , y)
    w , b = train_logistic_regression(X_train, y_train)
    y_pred = predict(X_test, w, b)
    print(accuracy(y_test,y_pred))

if __name__ == "__main__":
    main()
