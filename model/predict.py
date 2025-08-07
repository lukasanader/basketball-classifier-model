import pandas as pd
import joblib
from datetime import datetime
from nba_api.stats.endpoints import teamgamelog
import time

def get_latest_team_stats(team_id, num_games=5):
    """Fetch the last `num_games` for a team and calculate rolling averages + rest days."""
    time.sleep(1)  # Sleep to avoid rate limiting
    try:
        log = teamgamelog.TeamGameLog(team_id=team_id, season='2024-25')
        df = log.get_data_frames()[0].head(num_games)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')
        df = df.sort_values('GAME_DATE')

        stats = {
            'DAYS_SINCE_LAST_GAME': (datetime.now() - df['GAME_DATE'].max()).days,
            'PTS_ROLL5': df['PTS'].mean(),
            'FG_PCT_ROLL5': df['FG_PCT'].mean(),
            'REB_ROLL5': df['REB'].mean(),
            'AST_ROLL5': df['AST'].mean(),
            'TOV_ROLL5': df['TOV'].mean()
        }
        return stats
    except Exception as e:
        print(f"Error fetching data for team {team_id}: {e}")
        return None

def build_feature_row(home_id, away_id, feature_columns):
    """Build the feature row for prediction using team IDs."""
    home_stats = get_latest_team_stats(home_id)
    away_stats = get_latest_team_stats(away_id)
    
    if not home_stats or not away_stats:
        return None

    game_date = datetime.now()

    row = {
        'IS_HOME_HOME': 1,
        'DAYS_SINCE_LAST_GAME_HOME': home_stats['DAYS_SINCE_LAST_GAME'],
        'PTS_ROLL5_HOME': home_stats['PTS_ROLL5'],
        'FG_PCT_ROLL5_HOME': home_stats['FG_PCT_ROLL5'],
        'REB_ROLL5_HOME': home_stats['REB_ROLL5'],
        'AST_ROLL5_HOME': home_stats['AST_ROLL5'],
        'TOV_ROLL5_HOME': home_stats['TOV_ROLL5'],
        'TEAM_ID_HOME': home_id,

        'IS_HOME_AWAY': 0,
        'DAYS_SINCE_LAST_GAME_AWAY': away_stats['DAYS_SINCE_LAST_GAME'],
        'PTS_ROLL5_AWAY': away_stats['PTS_ROLL5'],
        'FG_PCT_ROLL5_AWAY': away_stats['FG_PCT_ROLL5'],
        'REB_ROLL5_AWAY': away_stats['REB_ROLL5'],
        'AST_ROLL5_AWAY': away_stats['AST_ROLL5'],
        'TOV_ROLL5_AWAY': away_stats['TOV_ROLL5'],
        'TEAM_ID_AWAY': away_id,

        'GAME_DAYOFWEEK': game_date.weekday(),
        'GAME_MONTH': game_date.month,
        'GAME_DAY': game_date.day,
        'GAME_YEAR': game_date.year,
        'GAME_DAYOFYEAR': game_date.timetuple().tm_yday,
    }

    # Initialize DataFrame
    features = pd.DataFrame([row])

    # Ensure all expected columns are present
    for col in feature_columns:
        if col not in features.columns:
            features[col] = 0  # default value

    return features[feature_columns]


def get_team_id(team_name):
    TEAM_NAME_TO_ID = {
        'Atlanta Hawks': 1610612737, 'Boston Celtics': 1610612738, 'Brooklyn Nets': 1610612751,
        'Charlotte Hornets': 1610612766, 'Chicago Bulls': 1610612741, 'Cleveland Cavaliers': 1610612739,
        'Dallas Mavericks': 1610612742, 'Denver Nuggets': 1610612743, 'Detroit Pistons': 1610612765,
        'Golden State Warriors': 1610612744, 'Houston Rockets': 1610612745, 'Indiana Pacers': 1610612754,
        'LA Clippers': 1610612746, 'Los Angeles Lakers': 1610612747, 'Memphis Grizzlies': 1610612763,
        'Miami Heat': 1610612748, 'Milwaukee Bucks': 1610612749, 'Minnesota Timberwolves': 1610612750,
        'New Orleans Pelicans': 1610612740, 'New York Knicks': 1610612752, 'Oklahoma City Thunder': 1610612760,
        'Orlando Magic': 1610612753, 'Philadelphia 76ers': 1610612755, 'Phoenix Suns': 1610612756,
        'Portland Trail Blazers': 1610612757, 'Sacramento Kings': 1610612758, 'San Antonio Spurs': 1610612759,
        'Toronto Raptors': 1610612761, 'Utah Jazz': 1610612762, 'Washington Wizards': 1610612764
    }
    return TEAM_NAME_TO_ID.get(team_name)

def predict_game(home_team_name, away_team_name, model_path='nba_model.pkl'):
    home_team_id = get_team_id(home_team_name)
    away_team_id = get_team_id(away_team_name)

    if home_team_id is None or away_team_id is None:
        print("Invalid team name(s).")
        return

    # Load model and expected features
    model = joblib.load(model_path)
    feature_columns = joblib.load('feature_columns.pkl')
    print("Model trained with:", model.feature_names_in_)
    # Generate raw feature row
    raw_features = build_feature_row(home_team_id, away_team_id, feature_columns)
    if raw_features is None:
        print("Could not fetch stats or build input row.")
        return

    # Start fresh with only the expected columns
    features = pd.DataFrame(columns=feature_columns)

    # Copy over valid columns from raw_features
    for col in feature_columns:
        features.at[0, col] = raw_features[col].values[0] if col in raw_features.columns else 0

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    print("\nPrediction Result:")
    print(f"{home_team_name} vs. {away_team_name}")
    print(f"Predicted Winner: {'HOME (' + home_team_name + ')' if prediction == 1 else 'AWAY (' + away_team_name + ')'}")
    print(f"Home Win Probability: {probability:.2%}")


if __name__ == "__main__":
    home_team = input("Enter home team name: ")
    away_team = input("Enter away team name: ")
    predict_game(home_team, away_team)