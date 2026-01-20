import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data():
    local_train = "train.csv"
    local_test = "test.csv"
    kaggle_train = "/kaggle/input/titanic/train.csv"
    kaggle_test = "/kaggle/input/titanic/test.csv"

    if os.path.exists(local_train) and os.path.exists(local_test):
        train_path, test_path = local_train, local_test
    elif os.path.exists(kaggle_train) and os.path.exists(kaggle_test):
        train_path, test_path = kaggle_train, kaggle_test
    else:
        return None, None

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def preprocess(train_data, test_data):
    features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked", "Age"]

    train = train_data.copy()
    test = test_data.copy()

    train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
    test["Embarked"].fillna(test["Embarked"].mode()[0], inplace=True)

    train["Age"].fillna(train["Age"].median(), inplace=True)
    test["Age"].fillna(test["Age"].median(), inplace=True)

    X = pd.get_dummies(train[features])
    X_test = pd.get_dummies(test[features])

    X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

    y = train["Survived"]
    return X, y, X_test, test


def train_and_evaluate(X, y, random_state=1):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    return model, val_accuracy


def generate_submission(model, X_test, test_data, output_path="submission.csv"):
    predictions = model.predict(X_test)
    output = pd.DataFrame(
        {"PassengerId": test_data["PassengerId"], "Survived": predictions}
    )
    output.to_csv(output_path, index=False)
    return output_path


def main():
    train_data, test_data = load_data()

    if train_data is None or test_data is None:
        print(
            "No Titanic CSV files found. Place train.csv and test.csv here or run on Kaggle."
        )
        return

    X, y, X_test, test_df = preprocess(train_data, test_data)
    model, val_accuracy = train_and_evaluate(X, y)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    submission_path = generate_submission(model, X_test, test_df)
    print(f"Submission file saved to: {submission_path}")


if __name__ == "__main__":
    main()

