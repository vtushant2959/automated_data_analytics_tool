import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

# Function to analyze data
def analyze_data(data):
    print("Data Summary:")
    print(data.describe())
    print("\nData Info:")
    print(data.info())

# Function to visualize data
def visualize_data(data):
    # Check if data has at least two columns
    if len(data.columns) < 2:
        print("Data must have at least two columns for visualization.")
        return

    # Simple histogram
    plt.figure(figsize=(10,6))
    sns.histplot(data.iloc[:, 0], kde=True)
    plt.title('Histogram of First Column')
    plt.show()

    # Scatter plot
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1])
    plt.title('Scatter Plot of First Two Columns')
    plt.show()

# Function for simple machine learning
def simple_ml(data):
    # Check if data has at least two columns
    if len(data.columns) < 2:
        print("Data must have at least two columns for machine learning.")
        return

    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Check if data is suitable for regression
    if not all(isinstance(x, (int, float)) for x in y):
        print("Target variable must be numeric for regression.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

# Main function
def main():
    file_path = r'C:\Users\vtush\Desktop\All projects\Python\Automated_data_analytics_tool\Book1.csv'  # Use raw string for Windows paths
    data = load_data(file_path)

    if data is not None:
        analyze_data(data)
        visualize_data(data)
        simple_ml(data)

if __name__ == "__main__":
    main()
