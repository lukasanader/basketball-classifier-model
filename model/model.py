import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_dataset(path='nba_game_dataset.csv'):
    df = pd.read_csv(path)
    return df

def train_model(df):
    # Define target and features
    y = df['HOME_WIN']
    feature_cols = [col for col in df.columns if
                    col not in ['GAME_ID', 'TEAM_ID_HOME', 'TEAM_ID_AWAY',
                                'TEAM_ABBREVIATION_HOME', 'TEAM_ABBREVIATION_AWAY',
                                'PTS_HOME', 'PTS_AWAY', 'POINT_DIFF', 'HOME_WIN']]
    X = df[feature_cols]

    # Filter explicitly
    train_df = df[df['GAME_YEAR'] < 2025]   # Use past seasons for training
    test_df = df[df['GAME_YEAR'] == 2025]   # Use current season for testing

    # Features to always exclude
    leakage_cols = [
    'REB_HOME', 'AST_HOME', 'TOV_HOME',
    'REB_AWAY', 'AST_AWAY', 'TOV_AWAY',
    'MIN_HOME', 'MIN_AWAY',  # if present
    'FGM_HOME', 'FGM_AWAY', 'FGA_HOME', 'FGA_AWAY',
    'FTM_HOME', 'FTM_AWAY', 'FTA_HOME', 'FTA_AWAY',
    'FG_PCT_HOME', 'FG_PCT_AWAY', 'FT_PCT_HOME', 'FT_PCT_AWAY',
    'FG3M_HOME', 'FG3A_HOME', 'FG3_PCT_HOME',
    'FG3M_AWAY', 'FG3A_AWAY', 'FG3_PCT_AWAY',
    'PLUS_MINUS_HOME', 'PLUS_MINUS_AWAY',
    'PTS_HOME', 'PTS_AWAY', 'POINT_DIFF',
    'TEAM_ID_HOME', 'TEAM_ID_AWAY',
    'TEAM_ABBREVIATION_HOME', 'TEAM_ABBREVIATION_AWAY',
    'GAME_ID', 'HOME_WIN','OREB_HOME', 'DREB_HOME', 'STL_HOME', 'BLK_HOME', 'PF_HOME',
    'OREB_AWAY', 'DREB_AWAY', 'STL_AWAY', 'BLK_AWAY', 'PF_AWAY',
    ]

    feature_cols = [col for col in df.columns if col not in leakage_cols]
    X = X[feature_cols]


    X_train = train_df[feature_cols]
    y_train = train_df['HOME_WIN']

    X_test = test_df[feature_cols]
    y_test = test_df['HOME_WIN']
    print("Training features used:", feature_cols)

    # Train model
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(feature_cols, 'feature_columns.pkl')
    # Predictions
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Away Win', 'Home Win'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix: Home Win vs Away Win")
    plt.show()
    return model, X_train.columns, model.feature_importances_

def plot_feature_importance(features, importances, top_n=20):
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    imp_df = imp_df.sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=imp_df, palette='Blues_r')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = load_dataset()
    model, features, importances = train_model(df)
    # save the model
    joblib.dump(model, 'nba_model.pkl')
    plot_feature_importance(features, importances)
