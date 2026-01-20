# Titanic: Machine Learning from Disaster

This repository contains a minimal end‑to‑end solution for the classic Kaggle competition **Titanic: Machine Learning from Disaster**. It is designed to match the core ideas in your module:

- Steps in a typical machine learning workflow
- Training error vs. generalization error
- Bias–variance trade‑off
- Overfitting and how to control model complexity

All code lives in [titanic_rf_solution.py](file:///c:/Users/Yash/VS_PROJECTS/ML_Practical/Titanic-ML-Submission/titanic_rf_solution.py).

## Files

- `titanic_rf_solution.py` – Random Forest solution script (data loading, preprocessing, training, evaluation, and submission file generation).
- `README.md` – Explanation of the workflow and how it ties to ML concepts in the syllabus.
- `train.csv`, `test.csv` – Kaggle Titanic data files (you download these from the competition page and place them in this folder, or rely on Kaggle’s `/kaggle/input` paths when running as a notebook).

## 1. Workflow Overview

The solution follows the standard supervised learning pipeline:

1. **Data loading**
2. **Preprocessing and feature engineering**
3. **Train/validation split**
4. **Model training**
5. **Evaluation on a validation set**
6. **Training on full data and prediction on test set**
7. **Saving `submission.csv` for Kaggle**

### 1.1 Data Loading

The script first tries to load the data from the local folder:

- `train.csv`
- `test.csv`

If these are not found, it looks for the Kaggle notebook paths:

- `/kaggle/input/titanic/train.csv`
- `/kaggle/input/titanic/test.csv`

If neither is available, the script prints a message and exits cleanly. This keeps the code portable between your laptop and Kaggle.

### 1.2 Preprocessing and Feature Engineering

We focus on a small, high‑signal feature set:

- `Pclass`
- `Sex`
- `SibSp`
- `Parch`
- `Embarked`
- `Age`

Handling missing values:

- `Embarked` is filled with the most frequent value (mode).
- `Age` is filled with the median age.

Categorical variables (`Sex`, `Embarked`) are converted to numeric form using one‑hot encoding (`pandas.get_dummies`). After encoding, we align the train and test matrices so they have the same columns, which is critical when some categories appear only in one split.

### 1.3 Train/Validation Split

To estimate **generalization error**, we split the training data:

- `X_train`, `X_val`, `y_train`, `y_val` using an 80/20 split (`train_test_split` with `test_size=0.2` and `random_state=1`).

The model is trained on `X_train, y_train` and evaluated on `X_val, y_val`. The validation accuracy approximates how well the model will perform on truly unseen passengers.

## 2. Model: Random Forest Classifier

We use a **Random Forest Classifier**, which is well suited for mixed numeric and categorical data after encoding.

Key hyperparameters in this implementation:

- `n_estimators=100` – number of trees in the forest.
- `max_depth=5` – limits tree depth to control complexity.
- `random_state=1` – makes results reproducible.

The script:

- Fits the model on the training split.
- Predicts on the validation split.
- Prints the validation accuracy.

You can change `max_depth` or `n_estimators` to see how it affects performance and overfitting.

## 3. Generalization, Overfitting, and Bias–Variance

This project directly illustrates several module 1 concepts.

### 3.1 Training vs. Generalization Error

- **Training error** is how often the model misclassifies passengers in the data it was fit on (`X_train`, `y_train`). This is usually low, especially for flexible models like random forests.
- **Generalization error** is how often it misclassifies *new* passengers. We approximate this using the validation set (`X_val`, `y_val`), and we report it via the validation accuracy printed by the script.

If the training accuracy is high but validation accuracy is much lower, the model is overfitting.

### 3.2 Overfitting and Model Complexity

Random forests are ensembles of decision trees. If each tree is allowed to grow too deep:

- The model can memorize training examples.
- Training accuracy can reach 100%.
- Generalization error can increase because the model is too tailored to noise in the training data.

We control complexity here by using:

- A limited `max_depth=5` for each tree.
- Multiple trees (`n_estimators=100`) averaged together, which reduces variance.

You can experiment with:

- Larger `max_depth` → more complex trees, higher variance, more risk of overfitting.
- Smaller `max_depth` → simpler trees, higher bias, risk of underfitting if the model is too simple.

### 3.3 Bias–Variance Trade‑off

- **High bias (underfitting):** If you only used one or two simple features (for example, only `Sex`), the model might be too simple and miss important patterns. Training and validation accuracy would both be low.
- **High variance (overfitting):** If you let trees grow very deep (`max_depth=None`) with many features, the model may fit the training data extremely well but perform poorly on validation data.

A moderate `max_depth` and a carefully chosen feature set give a balance between bias and variance.

## 4. Generating `submission.csv`

Once satisfied with validation performance, the script:

1. Trains the Random Forest on the full training feature matrix `X, y`.
2. Uses the trained model to predict survival for the test passengers (`X_test`).
3. Writes a `submission.csv` file with two columns:
   - `PassengerId`
   - `Survived` (0 or 1 predictions)

You can upload this file directly to Kaggle under **“Submit Predictions”** on the competition page.

## 5. How to Run

### 5.1 Local Machine

1. Download `train.csv` and `test.csv` from the Kaggle Titanic competition.
2. Place both files in this `Titanic-ML-Submission` folder.
3. From this directory, run:

   ```bash
   python titanic_rf_solution.py
   ```

4. Check the printed validation accuracy.
5. Find `submission.csv` in the same folder and upload it to Kaggle.

### 5.2 Kaggle Notebook

1. Create a new Kaggle notebook attached to the Titanic competition.
2. Either:
   - Copy the contents of `titanic_rf_solution.py` into a code cell, or
   - Upload this script as a dataset and import it.
3. Run the cell. On Kaggle, the `/kaggle/input/titanic/` paths are available automatically.
4. When the notebook finishes, use the **“Submit Predictions”** button and select the generated `submission.csv`.

## 6. Suggested Report Points

For your course submission, you can briefly discuss:

- The full ML pipeline you implemented (loading → preprocessing → splitting → training → evaluating → submitting).
- How you handled missing data and categorical variables.
- How you measured generalization error using a validation set.
- How changing `max_depth` and `n_estimators` would move the model along the bias–variance spectrum and affect overfitting.

This gives a clean, well‑structured example of applying the core ideas from your module to a real Kaggle problem.

