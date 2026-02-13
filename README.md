# Chapter 9: Decision Trees

Implementation of decision tree algorithms for both classification and regression tasks using scikit-learn. This project demonstrates decision tree concepts including classification boundaries, overfitting prevention, and regression predictions.

## Repository Structure

```
Chapter-9-Decision-Trees/
│
├── data/
│   └── Admission_Predict.csv          # University admissions dataset
│
├── src/
│   ├── University_Admissions.py       # Decision tree classification script
│   └── Regression_decision_tree.py    # Decision tree regression script
│
├── README.md                           # Project documentation
├── LICENSE                             # MIT License
└── .gitignore                          # Git ignore rules
```

## Overview

This project explores decision trees through two distinct applications:

1. **University Admissions Classification** - Predicts whether a student will be admitted based on academic performance metrics (GRE, TOEFL, CGPA, etc.)
2. **Age-based Activity Regression** - Demonstrates decision tree regression using a simple synthetic dataset

## Dataset

### Admission_Predict.csv
Retrieved from http://mng.bz/aZlJ

**Features:**
- **GRE Score** - Graduate Record Examination score
- **TOEFL Score** - Test of English as a Foreign Language score
- **University Rating** - Rating of the university (1-5)
- **SOP** - Statement of Purpose strength (1-5)
- **LOR** - Letter of Recommendation strength (1-5)
- **CGPA** - Cumulative Grade Point Average
- **Research** - Research experience (0 or 1)
- **Chance of Admit** - Probability of admission (0-1)

For classification, the dataset creates binary labels using a 75% admission chance threshold.

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Chapter-9-Decision-Trees.git
cd Chapter-9-Decision-Trees
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

### University Admissions Classification

Navigate to the src directory and run:

```bash
cd src
python University_Admissions.py
```

**What it does:**
- Loads the admissions dataset
- Creates binary classification labels (admitted vs. not admitted)
- Trains three decision tree models with varying complexity:
  - **Full tree** - No constraints (demonstrates overfitting)
  - **Pruned tree** - Controlled depth and minimum samples
  - **Visualization tree** - Uses only GRE and TOEFL for 2D plotting
- Displays decision boundaries and tree structures
- Makes predictions on sample data

**Output:**
- Training accuracy scores
- Decision boundary visualizations
- Graphical tree structure displays
- Sample predictions

### Regression Decision Trees

Navigate to the src directory and run:

```bash
cd src
python Regression_decision_tree.py
```

**What it does:**
- Creates a simple age vs. activity frequency dataset
- Trains a decision tree regressor with max_depth=2
- Visualizes regression predictions
- Calculates Total Squared Error (TSE) for all possible splits
- Demonstrates how decision trees choose optimal split points

**Output:**
- Scatter plot of data points
- Regression prediction curve
- Tree structure visualization
- TSE calculations for each potential split

## Key Concepts Demonstrated

### Decision Tree Classification
- **Binary classification** using multiple features
- **Hyperparameter tuning** to prevent overfitting:
  - `max_depth` - Limits tree depth
  - `min_samples_leaf` - Minimum samples in leaf nodes
  - `min_samples_split` - Minimum samples to split a node
- **Decision boundaries** in 2D feature space
- **Tree visualization** for interpretability

### Decision Tree Regression
- **Continuous value prediction**
- **Total Squared Error (TSE)** as splitting criterion
- **Mean squared error** calculation
- **Regression line** visualization

### Overfitting Prevention
The project demonstrates three tree variants:

1. **Unpruned Tree** (`DecisionTreeClassifier()`)
   - No restrictions
   - Perfect training accuracy
   - Likely overfits the data

2. **Pruned Tree** (`max_depth=3, min_samples_leaf=10, min_samples_split=10`)
   - Balanced complexity
   - Better generalization
   - Lower training accuracy but more robust

3. **Simple Tree** (`max_depth=1` or `max_depth=2`)
   - Highly interpretable
   - May underfit complex patterns
   - Good for visualization

## Visualization Features

- **Decision Boundary Plots** - Shows how the tree partitions the feature space
- **Tree Structure Diagrams** - Displays nodes, splits, and leaf values
- **Scatter Plots** - Data points colored by class or showing regression targets
- **Regression Curves** - Smooth prediction lines over the input range

## Learning Outcomes

After working through this project, you will understand:

✅ How decision trees make predictions through recursive partitioning  
✅ The difference between classification and regression trees  
✅ How to prevent overfitting using hyperparameters  
✅ How to visualize decision boundaries and tree structures  
✅ How Total Squared Error (TSE) guides split selection  
✅ Trade-offs between model complexity and generalization  

## Example Predictions

```python
# High GRE & TOEFL, strong SOP
dt_smaller.predict([[320, 110, 3, 4.0, 3.5, 8.9, 0]])
# Output: [True] - Predicted: Admitted

# High GRE & TOEFL, weaker SOP
dt_smaller.predict([[320, 110, 3, 4.0, 3.5, 8.0, 0]])
# Output: [False] - Predicted: Not Admitted
```

## Notes

- The 75% admission threshold is arbitrary and used for demonstration purposes
- Decision trees without depth limits tend to memorize training data
- 2D visualizations use only GRE and TOEFL scores for plotting convenience
- The regression example uses synthetic data to clearly show splitting mechanics
- Run scripts from the `src/` directory to ensure correct relative paths

## References

- [Original Book Repository](https://github.com/luisguiserrano/manning)
- Dataset: Retrieved from http://mng.bz/aZlJ

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset sourced from Kaggle
- Based on examples from "Grokking Machine Learning" by Luis G. Serrano
