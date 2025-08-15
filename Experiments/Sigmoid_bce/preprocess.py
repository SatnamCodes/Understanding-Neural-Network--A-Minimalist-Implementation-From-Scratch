import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_preprocess_data(path):
    # Load the dataset
    df = pd.read_csv(path)

    # Drop unnecessary columns
    df.drop(columns=["id", "unnamned: 32"], inplace=True, errors="ignore")
    df.dropna(axis=1, thresh=len(df) * 0.5, inplace=True)
    df.dropna(inplace=True)


    # Encode target labels: M = 1, B = 0
    df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])

    # Split features and labels
    X = df.drop('diagnosis', axis=1).values
    y = df['diagnosis'].values.reshape(-1, 1)

    # Train-test split
    x_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Check for NaNs or infinite values in X/y before scaling
    print("Checking X_train for NaNs or infinite values:")
    print(pd.DataFrame(x_train).isnull().sum())
    print(np.isfinite(x_train).all())

    print("Checking y_train for NaNs or infinite values:")
    print(pd.DataFrame(y_train).isnull().sum())
    print(np.isfinite(y_train).all())

    # Normalize the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(X_test)

    return x_train, X_test, y_train, y_test
