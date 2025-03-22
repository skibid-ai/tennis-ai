#!/usr/bin/env python3
import json
import subprocess
import sys

def test_examples():
    """Test the tennis predictor with some example matches."""
    
    # Example matches to test
    test_matches = [
        {"playerOne": "Federer", "playerTwo": "Nadal", "surface": "Clay"},
        {"playerOne": "Djokovic", "playerTwo": "Murray", "surface": "Hard"},
        {"playerOne": "Alcaraz", "playerTwo": "Sinner", "surface": "Hard"},
        {"playerOne": "Medvedev", "playerTwo": "Zverev"}  # Default surface (Hard)
    ]
    
    print("Testing Tennis Match Predictor\n")
    
    for i, match in enumerate(test_matches, 1):
        print(f"Test {i}: {match['playerOne']} vs {match['playerTwo']} on {match.get('surface', 'Hard')}")
        
        # Convert match data to JSON string
        match_json = json.dumps(match)
        
        try:
            # Run the predictor script with the match data
            result = subprocess.run(
                [sys.executable, "main.py", match_json],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse and display the result
            prediction = json.loads(result.stdout)
            if "prediction" in prediction:
                winner = prediction["prediction"]["winner"]
                confidence = prediction["prediction"]["confidence"]
                print(f"  Predicted winner: {winner} (confidence: {confidence})")
                print(f"  {match['playerOne']} stats: win rate={prediction['prediction']['player1_stats']['win_rate']}, "
                      f"surface win rate={prediction['prediction']['player1_stats']['surface_win_rate']}")
                print(f"  {match['playerTwo']} stats: win rate={prediction['prediction']['player2_stats']['win_rate']}, "
                      f"surface win rate={prediction['prediction']['player2_stats']['surface_win_rate']}")
            else:
                print(f"  Error: {prediction.get('error', 'Unknown error')}")
        except subprocess.CalledProcessError as e:
            print(f"  Error executing prediction: {e}")
            if e.stderr:
                print(f"  Details: {e.stderr}")
        except json.JSONDecodeError:
            print("  Error parsing prediction result")
        
        print()

if __name__ == "__main__":
    test_examples() 