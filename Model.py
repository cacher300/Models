from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def train_model(data_file):
    # Load the data
    mcDavid_data = pd.read_csv(data_file)

    # Preprocess the data
    mcDavid_data['opp_strength'] = mcDavid_data['opp_ga_per_gp']
    X = mcDavid_data[['goals', 'shots', 'shooting_pct', 'opp_strength']]
    y = mcDavid_data['scored']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the data
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.transform(X_test)

    # Fit a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f'Train accuracy: {train_score:.4f}')
    print(f'Test accuracy: {test_score:.4f}')

    return model, scaler


def predict_odds(model, scaler, gpg, shots, shooting_pct, opp_gaa, data_file):
    # Load the data
    mcDavid_data = pd.read_csv(data_file)

    # Preprocess the data
    mcDavid_data['opp_strength'] = mcDavid_data['opp_ga_per_gp']
    X = mcDavid_data[['goals', 'shots', 'shooting_pct', 'opp_strength']]
    y = mcDavid_data['scored']

    # Scale the data
    X_new = np.array([[gpg, shots, shooting_pct, opp_gaa]])
    X_new_scaled = scaler.transform(X_new)

    # Make predictions for a hypothetical game
    odds = model.predict_proba(X_new_scaled)[0][1]

    return odds
