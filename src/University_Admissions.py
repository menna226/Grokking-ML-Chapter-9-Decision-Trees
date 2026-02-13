import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

np.random.seed(0)  # for reproducibility

# -------------------------------
# Function to plot data points
def plot_points(features, labels, size_of_points=100):
    X = np.array(features)
    Y = np.array(labels)
    # Separate spam and ham points
    spam = X[np.argwhere(Y==1)]
    ham = X[np.argwhere(Y==0)]
    plt.scatter([s[0][0] for s in spam],
                [s[0][1] for s in spam],
                s = size_of_points,
                color = 'cyan',
                edgecolor = 'k',
                marker = '^')
    plt.scatter([s[0][0] for s in ham],
                [s[0][1] for s in ham],
                s = size_of_points,
                color = 'red',
                edgecolor = 'k',
                marker = 's')

# -------------------------------
# Function to plot the decision boundary of the model
def plot_model(X, y, model, size_of_points=100):
    X = np.array(X)
    y = np.array(y)
    plot_step = 0.2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, colors=['red', 'blue'], alpha=0.2, levels=range(-1,2))
    plt.contour(xx, yy, Z,colors = 'k',linewidths = 1)
    plot_points(X, y, size_of_points)
    plt.show()
    
# -------------------------------
# Function to display the decision tree graphically
def display_tree(dt):
    from sklearn import tree
    plt.figure(figsize=(12,8))
    tree.plot_tree(dt, feature_names=features.columns, filled=True)
    plt.show()
    
# -------------------------------
# Load dataset and create labels
data = pd.read_csv('../data/Admission_Predict.csv', index_col=0)
data.columns = data.columns.str.strip()  # remove whitespace
data['Admitted'] = data['Chance of Admit'] >= 0.75  # classify high chance as admitted
data = data.drop(['Chance of Admit'], axis=1)

features = data.drop(['Admitted'], axis=1)
labels = data['Admitted']

# -------------------------------
# Train a decision tree classifier
dt = DecisionTreeClassifier()
dt.fit(features, labels)
print("Predictions for first 5 rows:", dt.predict(features[0:5]))
print("Training accuracy:", dt.score(features, labels))
display_tree(dt)

# -------------------------------
# Train a smaller tree to avoid overfitting
dt_smaller = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, min_samples_split=10)
dt_smaller.fit(features, labels)
print("Smaller tree training accuracy:", dt_smaller.score(features, labels))
display_tree(dt_smaller)

# Predict some sample data
print("Prediction for high GRE & TOEFL:", dt_smaller.predict([[320,110,3,4.0,3.5,8.9,0]]))
print("Prediction for lower SOP:", dt_smaller.predict([[320,110,3,4.0,3.5,8.0,0]]))

# -------------------------------
# Train trees using only GRE and TOEFL features for visualization
exams = data[['GRE Score', 'TOEFL Score']]
plot_points(exams, labels, size_of_points=25)

dt_exams = DecisionTreeClassifier(max_depth=2)
dt_exams.fit(exams, labels)
plot_model(exams, labels, dt_exams, size_of_points=25)
display_tree(dt_exams)

simpler_dt_exams = DecisionTreeClassifier(max_depth=1)
simpler_dt_exams.fit(exams, labels)
plot_model(exams, labels, simpler_dt_exams, size_of_points=25)
display_tree(simpler_dt_exams)

crazy_dt_exams = DecisionTreeClassifier()
crazy_dt_exams.fit(exams, labels)
plot_model(exams, labels, crazy_dt_exams, size_of_points=25)
display_tree(crazy_dt_exams)