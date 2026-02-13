import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# -------------------------------
# Function to display the decision tree graphically
def display_tree(dt):
    from sklearn import tree
    plt.figure(figsize=(12,8))
    # feature_names=["Age"] because we only have one feature here
    tree.plot_tree(dt, feature_names=["Age"], filled=True)
    plt.show()

# -------------------------------
# Function to plot the regression predictions along with actual points
def plot_regressor(model, features, labels):
    x = np.linspace(0,85,1000)  # create smooth x-axis for plotting model predictions
    plt.scatter(features, labels)  # actual data points
    plt.plot(x, model.predict(x.reshape([-1,1])))  # regression model predictions
    plt.xlabel("Age")
    plt.ylabel("Days per week")
    plt.show()


# -------------------------------
# Sample dataset: Age vs Days per week doing an activity
features = [[10],[20],[30],[40],[50],[60],[70],[80]]
labels = [7,5,7,1,2,1,5,4]

plt.scatter(features, labels)
plt.xlabel("Age")
plt.ylabel("Days per week")
plt.show()

# -------------------------------
# Train a decision tree regressor with max depth 2
dt_regressor = DecisionTreeRegressor(max_depth=2)
dt_regressor.fit(features, labels)

display_tree(dt_regressor)  # show tree structure
plot_regressor(dt_regressor, features, labels)  # plot predictions

# -------------------------------
# Manual calculation of split errors (TSE) to understand tree splits
for i in range(0,9):
    left = np.array(labels[:i])
    right = np.array(labels[i:])
    print("****** Split at index", i, "******")
    print("Left:", left, "Right:", right)
    print("Mean Left:", np.mean(left), "Mean Right:", np.mean(right))
    # Total squared error (TSE)
    left_tse = left - np.mean(left)
    right_tse = right - np.mean(right)
    print("Split TSE:", 1/8*(np.dot(left_tse, left_tse) + np.dot(right_tse, right_tse)))

# Overall mean of all labels
print("Overall mean:", np.array([7,5,7,1,2,1,5,4]).mean())