from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
from time import sleep

def fetch_last_n_seasons(n=8):
    all_games = []
    current_year = 2025 # adjust based on today's year
    for i in range(n):
        start = current_year - i - 1
        end = str(start + 1)[-2:]
        season_str = f"{start}-{end}"
        print(f"Fetching season {season_str}...")
        try:
            finder = leaguegamefinder.LeagueGameFinder(season_nullable=season_str)
            df = finder.get_data_frames()[0]
            df['SEASON'] = season_str
            all_games.append(df)
            sleep(1)  # avoid rate limiting
        except Exception as e:
            print(f"Failed to fetch {season_str}: {e}")
    
    return pd.concat(all_games, ignore_index=True)

# Save to CSV
df = fetch_last_n_seasons(n=8) 
df.to_csv('data/games.csv', index=False)
