# Alphabet Character Recognition (Machine Learning)

This project explores the effectiveness of various machine learning models, specifically K-Nearest Neighbors (KNN) and Decision Trees, to identify handwritten characters from the English TypeAlphabet dataset. The project demonstrates an end-to-end pipeline from data preprocessing to hyperparameter optimization.

## 🚀 Technologies Used
* **Language:** Python
* **ML Libraries:** `scikit-learn`, `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`

## 🧠 Key Experiments
* **Data Preprocessing:** Handled an image dataset of 26,416 entries, performing dimensionality reduction by focusing on relevant pixel attributes (28x28 matrix normalization).
* **Binary Classification:** Implemented a KNN model to distinguish between characters 'O' and 'L'. Achieved an accuracy of 0.995 with K=1 and a selected range of 15 attributes.
* **Multiclass Classification:** Built and optimized Decision Tree models to classify the entire English alphabet.
* **Performance Metrics:** Utilized K-Fold Cross-Validation (5 folds) and analyzed Confusion Matrices to identify model limitations and common classification errors (e.g., misclassifying 'Y' as 'T').

## 🛠 Methodology
1. **Exploratory Data Analysis (EDA):** Visualized pixel intensity (heatmaps) to identify the most relevant attributes for classification.
2. **Hyperparameter Tuning:** Systematically varied tree depth, impurity criteria (Gini vs. Entropy), and the number of attributes to maximize performance.
3. **Model Selection:** Selected the optimal model based on an average accuracy of 0.7733 in cross-validation, achieving 0.7627 accuracy on the held-out test set.

## 📈 Key Visualizations
*(Insert screenshots of your best charts here, such as the Confusion Matrix or the Decision Tree Accuracy vs. Depth graph)*

![Confusion Matrix](path/to/your/confusion_matrix.png)

## ⚙️ How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/elciclon/your-repository.git](https://github.com/elciclon/your-repository.git)
