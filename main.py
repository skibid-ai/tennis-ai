import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json
import sys

def load_data():
    """Load and preprocess the tennis match data."""
    print("Loading tennis match data...")
    df = pd.read_csv('atp_tennis.csv')
    
    # Clean and prepare data
    df = df.dropna(subset=['Player_1', 'Player_2', 'Winner', 'Rank_1', 'Rank_2', 'Surface'])
    
    # Extract last names from player names (assuming format is "Last N.")
    df['Player_1_Last'] = df['Player_1'].apply(lambda x: x.split()[0] if pd.notna(x) else x)
    df['Player_2_Last'] = df['Player_2'].apply(lambda x: x.split()[0] if pd.notna(x) else x)
    df['Winner_Last'] = df['Winner'].apply(lambda x: x.split()[0] if pd.notna(x) else x)
    
    # Convert ranks to numeric, handling any errors
    df['Rank_1'] = pd.to_numeric(df['Rank_1'], errors='coerce')
    df['Rank_2'] = pd.to_numeric(df['Rank_2'], errors='coerce')
    
    # Create the target variable: 1 if Player_1 wins, 0 if Player_2 wins
    df['Player_1_Win'] = (df['Winner_Last'] == df['Player_1_Last']).astype(int)
    
    return df

def extract_features(df):
    """Extract relevant features for prediction."""
    print("Extracting features...")
    
    # Encode surface categories
    surface_encoder = LabelEncoder()
    df['Surface_Encoded'] = surface_encoder.fit_transform(df['Surface'])
    
    # Create player statistics dictionaries
    player_stats = {}
    
    # Group by player and surface to get win rates
    for player in set(df['Player_1_Last'].tolist() + df['Player_2_Last'].tolist()):
        # Matches where player was Player_1
        p1_matches = df[df['Player_1_Last'] == player]
        p1_wins = p1_matches[p1_matches['Player_1_Win'] == 1].shape[0]
        
        # Matches where player was Player_2
        p2_matches = df[df['Player_2_Last'] == player]
        p2_wins = p2_matches[p2_matches['Player_1_Win'] == 0].shape[0]
        
        total_matches = p1_matches.shape[0] + p2_matches.shape[0]
        total_wins = p1_wins + p2_wins
        
        if total_matches > 0:
            win_rate = total_wins / total_matches
        else:
            win_rate = 0.5  # Default for players with no history
        
        # Surface-specific win rates
        surface_win_rates = {}
        for surface in df['Surface'].unique():
            p1_surface_matches = p1_matches[p1_matches['Surface'] == surface]
            p1_surface_wins = p1_surface_matches[p1_surface_matches['Player_1_Win'] == 1].shape[0]
            
            p2_surface_matches = p2_matches[p2_matches['Surface'] == surface]
            p2_surface_wins = p2_surface_matches[p2_surface_matches['Player_1_Win'] == 0].shape[0]
            
            total_surface_matches = p1_surface_matches.shape[0] + p2_surface_matches.shape[0]
            total_surface_wins = p1_surface_wins + p2_surface_wins
            
            if total_surface_matches > 0:
                surface_win_rates[surface] = total_surface_wins / total_surface_matches
            else:
                surface_win_rates[surface] = 0.5
        
        # Get average rank
        p1_ranks = p1_matches['Rank_1'].mean() if not p1_matches.empty else None
        p2_ranks = p2_matches['Rank_2'].mean() if not p2_matches.empty else None
        
        if p1_ranks is not None and p2_ranks is not None:
            avg_rank = (p1_ranks + p2_ranks) / 2
        elif p1_ranks is not None:
            avg_rank = p1_ranks
        elif p2_ranks is not None:
            avg_rank = p2_ranks
        else:
            avg_rank = 100  # Default rank if unknown
        
        player_stats[player] = {
            'win_rate': win_rate,
            'surface_win_rates': surface_win_rates,
            'avg_rank': avg_rank,
            'match_count': total_matches
        }
    
    return df, player_stats, surface_encoder

def train_model(df):
    """Train a machine learning model on the historical match data."""
    print("Training model...")
    
    # Features for model training
    features = ['Rank_1', 'Rank_2', 'Surface_Encoded']
    X = df[features].copy()
    y = df['Player_1_Win']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def predict_winner(player1, player2, surface="Hard", player_stats=None, model=None, surface_encoder=None):
    """Predict the winner of a match between player1 and player2."""
    if player_stats is None or model is None or surface_encoder is None:
        return {"error": "Model or player statistics not available."}
    
    # Get player stats or use defaults if player not in database
    player1_stats = player_stats.get(player1, {
        'win_rate': 0.5,
        'surface_win_rates': {surface: 0.5},
        'avg_rank': 100,
        'match_count': 0
    })
    
    player2_stats = player_stats.get(player2, {
        'win_rate': 0.5,
        'surface_win_rates': {surface: 0.5},
        'avg_rank': 100,
        'match_count': 0
    })
    
    # Encode surface
    try:
        surface_encoded = surface_encoder.transform([surface])[0]
    except:
        # Default to hard court if surface not recognized
        hard_index = list(surface_encoder.classes_).index('Hard') if 'Hard' in surface_encoder.classes_ else 0
        surface_encoded = hard_index
    
    # Create feature vector for prediction
    X_pred = pd.DataFrame({
        'Rank_1': [player1_stats['avg_rank']],
        'Rank_2': [player2_stats['avg_rank']],
        'Surface_Encoded': [surface_encoded]
    })
    
    # Make prediction
    prob_player1_win = model.predict_proba(X_pred)[0][1]
    
    # Adjust prediction based on surface-specific win rates
    player1_surface_rate = player1_stats['surface_win_rates'].get(surface, 0.5)
    player2_surface_rate = player2_stats['surface_win_rates'].get(surface, 0.5)
    
    # Combine model prediction with surface history
    final_prob = 0.7 * prob_player1_win + 0.3 * (player1_surface_rate / (player1_surface_rate + player2_surface_rate))
    
    # Determine the winner
    if final_prob >= 0.5:
        winner = player1
        confidence = final_prob
    else:
        winner = player2
        confidence = 1 - final_prob
    
    return {
        "prediction": {
            "winner": winner,
            "confidence": round(float(confidence), 2),
            "player1_stats": {
                "win_rate": round(player1_stats['win_rate'], 2),
                "surface_win_rate": round(player1_stats['surface_win_rates'].get(surface, 0.5), 2),
                "avg_rank": int(player1_stats['avg_rank']),
                "matches_played": player1_stats['match_count']
            },
            "player2_stats": {
                "win_rate": round(player2_stats['win_rate'], 2),
                "surface_win_rate": round(player2_stats['surface_win_rates'].get(surface, 0.5), 2),
                "avg_rank": int(player2_stats['avg_rank']),
                "matches_played": player2_stats['match_count']
            }
        }
    }

def handle_input():
    """Process input from CLI or stdin."""
    if len(sys.argv) > 1:
        # Input from command line arguments
        try:
            input_data = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input."}
    else:
        # Input from stdin
        try:
            input_data = json.loads(sys.stdin.read())
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input."}
    
    # Extract player names and optional surface
    player1 = input_data.get("playerOne")
    player2 = input_data.get("playerTwo")
    surface = input_data.get("surface", "Hard")
    
    if not player1 or not player2:
        return {"error": "Both playerOne and playerTwo must be provided."}
    
    return {"player1": player1, "player2": player2, "surface": surface}

if __name__ == "__main__":
    # Load and prepare data
    df = load_data()
    df, player_stats, surface_encoder = extract_features(df)
    model = train_model(df)
    
    # Handle input
    input_result = handle_input()
    
    if "error" in input_result:
        print(json.dumps(input_result))
    else:
        # Make prediction
        result = predict_winner(
            input_result["player1"],
            input_result["player2"],
            input_result["surface"],
            player_stats,
            model,
            surface_encoder
        )
        
        # Output result as JSON
        print(json.dumps(result, indent=2))
