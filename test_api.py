#!/usr/bin/env python3
import requests
import json
import sys
import time
import subprocess
import signal
import os
from typing import Dict, List, Any

# API base URL
BASE_URL = "http://localhost:8004"

def wait_for_api(timeout=30):
    """Wait for the API to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200 and response.json().get("model_ready"):
                print("API is ready!")
                return True
            else:
                print("API is starting up, waiting...")
        except requests.exceptions.ConnectionError:
            print("Waiting for API to start...")
        time.sleep(2)
    return False

def test_health_endpoint():
    """Test the health endpoint."""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_surfaces_endpoint():
    """Test the surfaces endpoint."""
    print("\n=== Testing Surfaces Endpoint ===")
    response = requests.get(f"{BASE_URL}/surfaces")
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_players_endpoint():
    """Test the players endpoint."""
    print("\n=== Testing Players Endpoint ===")
    
    # Test without search parameter
    response = requests.get(f"{BASE_URL}/players?limit=5")
    print(f"Status code: {response.status_code}")
    print(f"Response (first 5 players): {json.dumps(response.json(), indent=2)}")
    
    # Test with search parameter
    if response.status_code == 200:
        # Get a player name from the first response to use in search
        players = response.json().get("players", [])
        if players:
            search_term = players[0]["name"][:3].lower()
            print(f"\nSearching for players with term: '{search_term}'")
            search_response = requests.get(f"{BASE_URL}/players?search={search_term}&limit=3")
            print(f"Status code: {search_response.status_code}")
            print(f"Response: {json.dumps(search_response.json(), indent=2)}")
    
    return response.status_code == 200

def test_prediction_endpoint():
    """Test the prediction endpoint with example matches."""
    print("\n=== Testing Prediction Endpoint ===")
    
    # Get some player names first
    try:
        response = requests.get(f"{BASE_URL}/players?limit=20")
        players = response.json().get("players", [])
        
        if len(players) < 2:
            print("Not enough players found to make predictions.")
            return False
        
        # Get surface options
        surfaces_response = requests.get(f"{BASE_URL}/surfaces")
        surfaces = surfaces_response.json().get("surfaces", ["Hard"])
        
        # Create some test matches
        test_matches = [
            {"playerOne": players[0]["name"], "playerTwo": players[1]["name"], "surface": surfaces[0]},
        ]
        
        # Add more matches if we have more players
        if len(players) > 3 and len(surfaces) > 1:
            test_matches.append({
                "playerOne": players[2]["name"], 
                "playerTwo": players[3]["name"], 
                "surface": surfaces[1]
            })
        
        # Test famous players if found
        famous_players = ["Federer", "Nadal", "Djokovic", "Murray"]
        found_players = []
        for famous in famous_players:
            for player in players:
                if famous.lower() in player["name"].lower():
                    found_players.append(player["name"])
                    if len(found_players) == 2:
                        break
            if len(found_players) == 2:
                break
        
        if len(found_players) == 2:
            test_matches.append({
                "playerOne": found_players[0],
                "playerTwo": found_players[1],
                "surface": "Clay" if "Clay" in surfaces else surfaces[0]
            })
        
        # Make predictions
        all_successful = True
        for i, match in enumerate(test_matches, 1):
            print(f"\nTest {i}: {match['playerOne']} vs {match['playerTwo']} on {match.get('surface', 'Hard')}")
            
            try:
                response = requests.post(f"{BASE_URL}/predict", json=match)
                print(f"Status code: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Predicted winner: {result['prediction']['winner']}")
                    print(f"Confidence: {result['prediction']['confidence']}")
                    print(f"{match['playerOne']} stats: {result['prediction']['player1_stats']}")
                    print(f"{match['playerTwo']} stats: {result['prediction']['player2_stats']}")
                else:
                    print(f"Error: {response.text}")
                    all_successful = False
                    
            except Exception as e:
                print(f"Error: {e}")
                all_successful = False
        
        return all_successful
    
    except Exception as e:
        print(f"Error setting up prediction tests: {e}")
        return False

def run_all_tests():
    """Run all API tests."""
    print("Starting API tests...")
    
    # Wait for API to start
    if not wait_for_api():
        print("API failed to start in time. Exiting.")
        return False
    
    # Run tests
    tests = [
        ("Health endpoint", test_health_endpoint),
        ("Surfaces endpoint", test_surfaces_endpoint),
        ("Players endpoint", test_players_endpoint),
        ("Prediction endpoint", test_prediction_endpoint)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        try:
            result = test_func()
            results.append(result)
            print(f"Test result: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"Test error: {e}")
            results.append(False)
    
    # Summary
    print("\n=== Test Summary ===")
    all_passed = all(results)
    for i, (test_name, _) in enumerate(tests):
        status = "PASS" if results[i] else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall result: {'All tests passed!' if all_passed else 'Some tests failed.'}")
    return all_passed

if __name__ == "__main__":
    # Check if the API is already running
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("API is already running. Running tests...")
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("API is not running. Starting API server...")
        
        # Start the API server
        api_process = subprocess.Popen(
            [sys.executable, "api.py"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            # Run tests
            success = run_all_tests()
            
            # Exit with appropriate code
            sys.exit(0 if success else 1)
            
        finally:
            # Terminate the API server
            print("\nShutting down API server...")
            if api_process:
                api_process.terminate()
                try:
                    api_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    api_process.kill()
            print("API server stopped.") 