import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df.sort_values(by=['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

def add_basic_features(df, rolling_window=5):
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs').astype(int)

    # Rest days
    df['PREV_GAME_DATE'] = df.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    df['DAYS_SINCE_LAST_GAME'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days
    df['DAYS_SINCE_LAST_GAME'] = df['DAYS_SINCE_LAST_GAME'].fillna(7)

    # Rolling averages
    rolling_cols = ['PTS', 'FG_PCT', 'REB', 'AST', 'TOV']
    for col in rolling_cols:
        df[f'{col}_ROLL{rolling_window}'] = (
            df.groupby('TEAM_ID')[col]
            .rolling(rolling_window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    return df

def merge_games(df):
    team_cols = [col for col in df.columns if col not in ['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION']]
    team_df = df[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"] + team_cols].copy()
    
    home_df = team_df[df['IS_HOME'] == 1].copy()
    away_df = team_df[df['IS_HOME'] == 0].copy()

    merged = pd.merge(
        home_df, away_df,
        on="GAME_ID",
        suffixes=('_HOME', '_AWAY')
    )

    merged['HOME_WIN'] = (merged['PTS_HOME'] > merged['PTS_AWAY']).astype(int)
    merged['POINT_DIFF'] = merged['PTS_HOME'] - merged['PTS_AWAY']
    return merged

def preprocess_data(path, rolling_window=5):
    df = load_data(path)
    df = add_basic_features(df, rolling_window)
    df = df.dropna()

    merged_df = merge_games(df)

    # Convert game date and extract features
    merged_df['GAME_DATE_HOME'] = pd.to_datetime(merged_df['GAME_DATE_HOME'])
    merged_df['GAME_DAYOFWEEK'] = merged_df['GAME_DATE_HOME'].dt.dayofweek
    merged_df['GAME_MONTH'] = merged_df['GAME_DATE_HOME'].dt.month
    merged_df['GAME_DAY'] = merged_df['GAME_DATE_HOME'].dt.day
    merged_df['GAME_YEAR'] = merged_df['GAME_DATE_HOME'].dt.year
    merged_df['GAME_DAYOFYEAR'] = merged_df['GAME_DATE_HOME'].dt.dayofyear

    # Drop string/text columns not suitable for modeling
    drop_cols = [
    'GAME_ID', 'TEAM_ABBREVIATION_HOME', 'TEAM_NAME_HOME',
    'GAME_DATE_HOME', 'MATCHUP_HOME', 'WL_HOME',
    'TEAM_ABBREVIATION_AWAY', 'TEAM_NAME_AWAY',
    'GAME_DATE_AWAY', 'MATCHUP_AWAY', 'WL_AWAY',
    'PREV_GAME_DATE_HOME', 'PREV_GAME_DATE_AWAY',
    'SEASON_ID_HOME', 'SEASON_ID_AWAY', 'SEASON_HOME', 'SEASON_AWAY'   

    ]

    merged_df = merged_df.drop(columns=[col for col in drop_cols if col in merged_df.columns])

    return merged_df

# Run preprocessing
final_df = preprocess_data('data/games.csv')
final_df.to_csv('nba_game_dataset.csv', index=False)
print("Preprocessing complete.")
