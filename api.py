from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import traceback
import time
import os
import uvicorn
import requests
from lxml import html
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Any

# Import functions from main.py
from main import load_data, extract_features, train_model, predict_winner

# Global variables to store model and data
model = None
player_stats = None
surface_encoder = None
model_ready = False
last_load_time = 0
MODEL_RELOAD_INTERVAL = 3600  # Reload model every hour

def initialize_model():
    """Load and initialize the model if not already loaded or if reload interval passed."""
    global model, player_stats, surface_encoder, model_ready, last_load_time
    
    current_time = time.time()
    if not model_ready or (current_time - last_load_time > MODEL_RELOAD_INTERVAL):
        try:
            print("Initializing model...")
            df = load_data()
            df, player_stats, surface_encoder = extract_features(df)
            model = train_model(df)
            model_ready = True
            last_load_time = current_time
            print("Model initialized successfully")
        except Exception as e:
            print(f"Error initializing model: {e}")
            traceback.print_exc()
            model_ready = False

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize model when the application starts
    initialize_model()
    yield
    # Shutdown: Clean up resources if needed
    # (No cleanup needed for this application)

# Create FastAPI instance with lifespan
app = FastAPI(
    title="Tennis Match Predictor API",
    description="API to predict tennis match outcomes based on historical data",
    version="1.0.0",
    lifespan=lifespan
)

# Define request and response models
class MatchPredictionRequest(BaseModel):
    playerOne: str = Field(..., description="Last name of the first player")
    playerTwo: str = Field(..., description="Last name of the second player")
    surface: Optional[str] = Field(None, description="Court surface (e.g., 'Hard', 'Clay', 'Grass'). If null, defaults to 'Hard'")

class PlayerStats(BaseModel):
    win_rate: float
    surface_win_rate: float
    avg_rank: int
    matches_played: int

class PredictionResult(BaseModel):
    winner: str
    confidence: float
    player1_stats: PlayerStats
    player2_stats: PlayerStats

class MatchPredictionResponse(BaseModel):
    prediction: PredictionResult

class PlayerInfo(BaseModel):
    name: str
    avg_rank: int
    win_rate: float
    matches_played: int

class PlayersResponse(BaseModel):
    players: List[PlayerInfo]

class SurfacesResponse(BaseModel):
    surfaces: List[str]

class HealthResponse(BaseModel):
    status: str
    model_ready: bool

class TournamentInfo(BaseModel):
    name: str
    dates: str
    tournament_id: Optional[str]
    original_url: str
    scoreboard_url: Optional[str]

class TournamentsResponse(BaseModel):
    tournament_count: int
    fetch_date: str
    tournaments: Dict[str, TournamentInfo]

class MatchPair(BaseModel):
    player1: str
    player2: str

class TournamentMatchesResponse(BaseModel):
    tournament_name: str
    tournament_day: str
    matches: Dict[str, MatchPair]

def get_model():
    """Dependency to ensure model is loaded."""
    if not model_ready:
        initialize_model()
        if not model_ready:
            raise HTTPException(status_code=500, detail="Model initialization failed")
    return {"model": model, "player_stats": player_stats, "surface_encoder": surface_encoder}

def get_current_tournaments():
    """
    Fetches the current tennis tournaments from the ESPN schedule page.
    
    Returns:
        dict: Dictionary with tournament information
    """
    url = "https://www.espn.com/tennis/schedule"
    
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # Get the page content
        print(f"Fetching tournaments from {url}...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the HTML with lxml
        tree = html.fromstring(response.content)
        
        # Find the section with current tournaments
        # Look for a title element that indicates "Current Tournaments"
        current_tournaments_section = None
        tournament_links = []  # Initialize this variable outside the conditional blocks
        
        section_headers = tree.xpath('//div[contains(@class, "Table__Title")]')
        
        for header in section_headers:
            if "Current" in header.text_content():
                # Found the Current Tournaments section
                current_tournaments_section = header.getparent().getparent()
                break
        
        if not current_tournaments_section:
            # Try alternative approach using the provided XPath pattern
            print("Trying alternative approach to find current tournaments...")
            tournament_links = tree.xpath('//*[@id="fittPageContainer"]/div[2]/div[2]/div/div/div[1]/section/div/div[4]/div/div/div[2]/div/div[2]/table/tbody/tr/td[2]/div/a')
            
            if not tournament_links:
                # One more attempt with a more general selector
                tournament_links = tree.xpath('//div[contains(@class, "Table__TBODY")]//a[contains(@href, "/tennis/")]')
        else:
            # Found the section, now get all tournament links within it
            tournament_links = current_tournaments_section.xpath('.//tbody//a[contains(@href, "/tennis/")]')
        
        if not tournament_links:
            print("No tournament links found.")
            return {}
        
        tournaments = {}
        for i, link in enumerate(tournament_links, 1):
            tournament_name = link.text_content().strip()
            href = link.get('href')
            
            # Skip player links - they contain "/player/" in the URL
            if href and "/player/" in href:
                continue
                
            # Skip links that don't contain "tournament" or "eventId" - they're likely not tournament links
            if href and not ("/tournament/" in href or "/eventId/" in href):
                continue
            
            # Try to find date information
            # This might be in a nearby cell
            tr_element = link.getparent()
            while tr_element is not None and tr_element.tag != 'tr':
                tr_element = tr_element.getparent()
            
            date_info = "Unknown Date"
            if tr_element is not None:
                date_cells = tr_element.xpath('./td[contains(@class, "date")]')
                if date_cells:
                    date_info = date_cells[0].text_content().strip()
            
            # Process the tournament URL to extract useful information
            tournament_id = None
            if href:
                # Extract tournament ID from URL if available
                id_match = re.search(r'eventId/([^/]+)', href)
                if id_match:
                    tournament_id = id_match.group(1)
            
            # Get today's date in the format YYYYMMDD for creating a scoreboard URL
            today = datetime.now().strftime("%Y%m%d")
            
            # Construct a scoreboard URL
            scoreboard_url = None
            if tournament_id:
                scoreboard_url = f"https://www.espn.com/tennis/scoreboard/tournament/_/eventId/{tournament_id}/competitionType/1/date/{today}"
            else:
                # If we couldn't extract a tournament ID but the URL contains "tournament", it's likely a valid tournament URL
                if href and "/tournament/" in href:
                    # Check if URL already has a date parameter
                    if "/date/" not in href:
                        # Add today's date to the URL
                        if href.endswith("/"):
                            scoreboard_url = f"{href}date/{today}"
                        else:
                            scoreboard_url = f"{href}/date/{today}"
                    else:
                        # URL already has a date, use it as is
                        scoreboard_url = href
            
            tournaments[f"tournament_{i}"] = {
                "name": tournament_name,
                "dates": date_info,
                "tournament_id": tournament_id,
                "original_url": href,
                "scoreboard_url": scoreboard_url
            }
        
        result = {
            "tournament_count": len(tournaments),
            "fetch_date": datetime.now().strftime("%Y-%m-%d"),
            "tournaments": tournaments
        }
        
        return result
        
    except Exception as e:
        print(f"Error accessing ESPN schedule: {e}")
        traceback.print_exc()
        return {"error": str(e), "tournament_count": 0, "tournaments": {}}

def get_tournament_matches(url):
    """
    Extracts player match pairs from a tournament scoreboard URL.
    
    Args:
        url (str): The ESPN tournament scoreboard URL
    
    Returns:
        dict: Dictionary with tournament information and match pairs
    """
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # Get the page content
        print(f"Fetching data from {url}...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the HTML with lxml
        tree = html.fromstring(response.content)
        
        # Extract tournament name using the specific XPath
        tournament_xpath = '//*[@id="fittPageContainer"]/div[2]/div[2]/div/div/div[1]/div/section/div/div[1]/div/h1'
        tournament_elements = tree.xpath(tournament_xpath)
        tournament_name = tournament_elements[0].text_content().strip() if tournament_elements else "Unknown Tournament"
        
        # Fallback to a more general selector if the specific XPath doesn't work
        if not tournament_elements:
            print("Tournament name not found with specific XPath, trying alternative...")
            tournament_elements = tree.xpath('//div[contains(@class, "ScoreboardHeader__Name")]')
            tournament_name = tournament_elements[0].text_content().strip() if tournament_elements else "Unknown Tournament"
            
        # Extract date from the URL (format YYYYMMDD)
        date_match = re.search(r'date/(\d{8})', url)
        date_str = "Unknown Date"
        if date_match:
            # Convert from YYYYMMDD to YYYY-MM-DD format
            date_raw = date_match.group(1)
            try:
                date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
            except Exception as e:
                print(f"Error formatting date: {e}")
                date_str = date_raw
        
        # Locate the main container first
        container_xpath = '//*[@id="fittPageContainer"]/div[2]/div[2]/div/div/div[1]/div/div/section/div/div[2]'
        container = tree.xpath(container_xpath)
        
        if not container:
            print("Main container not found. Website structure may have changed.")
            return {"error": "Main container not found"}
        
        # Find all div elements that have ul/li/a structure inside the container
        all_links = []
        
        # First approach: look for the pattern div/div/ul/li/a
        tabs_divs = tree.xpath(f"{container_xpath}/div")
        
        print(f"Found {len(tabs_divs)} potential tab divs")
        
        for div_index, div in enumerate(tabs_divs, 1):
            # Check if this div contains the list of links we're looking for
            link_elements = div.xpath('./div/ul/li/a')
            
            if not link_elements:
                # Try alternative structure
                link_elements = div.xpath('./ul/li/a')
            
            for li_index, link in enumerate(link_elements, 1):
                text = link.text_content().strip()
                href = link.get('href')
                
                all_links.append({
                    "section": div_index,
                    "index": li_index,
                    "text": text,
                    "href": href
                })
        
        # Now create player pairs from the list of player names
        player_names = [link['text'] for link in all_links]
        matches = {}
        
        # Create the base structure with tournament info
        result = {
            "Tournament Name": tournament_name,
            "Tournament Day": date_str
        }
        
        # Group players into pairs (matches)
        for i in range(0, len(player_names), 2):
            match_id = i // 2 + 1
            
            # Handle the case where there might be an odd number of players
            if i + 1 < len(player_names):
                result[f"match_{match_id}"] = {
                    "player1": player_names[i],
                    "player2": player_names[i + 1]
                }
            else:
                result[f"match_{match_id}"] = {
                    "player1": player_names[i],
                    "player2": "N/A"  # No opponent
                }
        
        return result
        
    except Exception as e:
        print(f"Error processing tournament data: {e}")
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", model_ready=model_ready)

@app.post("/predict", response_model=MatchPredictionResponse, tags=["Prediction"])
async def predict(data: MatchPredictionRequest, model_data: Dict = Depends(get_model)):
    """Predict the winner of a tennis match."""
    try:
        # Set default surface to 'Hard' if null
        surface = data.surface or "Hard"
        
        # Make prediction
        result = predict_winner(
            data.playerOne,
            data.playerTwo,
            surface,
            model_data["player_stats"],
            model_data["model"],
            model_data["surface_encoder"]
        )
        
        return result
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/players", response_model=PlayersResponse, tags=["Data"])
async def get_players(
    search: Optional[str] = Query(None, description="Search term to filter players by name"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results to return"),
    model_data: Dict = Depends(get_model)
):
    """Get available players in the dataset."""
    try:
        # Filter players by search term
        if search:
            search_lower = search.lower()
            matching_players = [
                player for player in model_data["player_stats"].keys() 
                if search_lower in player.lower()
            ]
        else:
            matching_players = list(model_data["player_stats"].keys())
        
        # Sort by match count (most experienced players first)
        sorted_players = sorted(
            matching_players,
            key=lambda p: model_data["player_stats"][p]['match_count'],
            reverse=True
        )[:limit]
        
        # Create result with player details
        result = []
        for player in sorted_players:
            stats = model_data["player_stats"][player]
            result.append(
                PlayerInfo(
                    name=player,
                    avg_rank=int(stats['avg_rank']),
                    win_rate=round(stats['win_rate'], 2),
                    matches_played=stats['match_count']
                )
            )
        
        return PlayersResponse(players=result)
        
    except Exception as e:
        print(f"Error retrieving players: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve players: {str(e)}")

@app.get("/surfaces", response_model=SurfacesResponse, tags=["Data"])
async def get_surfaces(model_data: Dict = Depends(get_model)):
    """Get available court surfaces."""
    try:
        surfaces = list(model_data["surface_encoder"].classes_)
        return SurfacesResponse(surfaces=surfaces)
    except Exception as e:
        print(f"Error retrieving surfaces: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve surfaces: {str(e)}")

@app.get("/tournaments", response_model=TournamentsResponse, tags=["Tournaments"])
async def get_tournaments():
    """
    Get current tennis tournaments from ESPN.
    
    Returns:
        Dictionary with tournament information
    """
    try:
        tournaments = get_current_tournaments()
        
        if "error" in tournaments:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve tournaments: {tournaments['error']}")
        
        return tournaments
    except Exception as e:
        print(f"Error retrieving tournaments: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tournaments: {str(e)}")

@app.get("/tournaments/{tournament_id}/matches", tags=["Tournaments"])
async def get_tournament_match_pairs(
    tournament_id: str, 
    date: Optional[str] = Query(None, description="Date in YYYYMMDD format. If not provided, today's date will be used."),
    predict: bool = Query(True, description="Whether to include match predictions"),
    model_data: Dict = Depends(get_model) if True else None  # Only get model data if predictions are requested
):
    """
    Get player pairs for a specific tournament with optional match predictions.
    
    Args:
        tournament_id: The ESPN tournament ID
        date: Optional date in YYYYMMDD format
        predict: Whether to include predictions for each match
        
    Returns:
        Dictionary with tournament information, match pairs, and predictions
    """
    try:
        # Get today's date if not provided
        if not date:
            date = datetime.now().strftime("%Y%m%d")
        
        # Construct URL
        url = f"https://www.espn.com/tennis/scoreboard/tournament/_/eventId/{tournament_id}/competitionType/1/date/{date}"
        
        # Get match pairs
        matches = get_tournament_matches(url)
        
        if "error" in matches:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve match pairs: {matches['error']}")
        
        # Format for consistency
        result = {
            "tournament_name": matches.get("Tournament Name", "Unknown Tournament"),
            "tournament_day": matches.get("Tournament Day", "Unknown Date"),
            "matches": {}
        }
        
        # Extract just the matches
        for key, value in matches.items():
            if key not in ["Tournament Name", "Tournament Day", "error"]:
                match_data = {
                    "player1": value["player1"],
                    "player2": value["player2"]
                }
                
                # Add prediction if requested and both players are valid
                if predict and model_ready and value["player2"] != "N/A":
                    try:
                        # Extract last names for prediction
                        player1_last = value["player1"].split()[-1]
                        player2_last = value["player2"].split()[-1]
                        
                        # Default to Hard court if we don't know
                        surface = "Hard"
                        
                        # Make prediction
                        prediction_result = predict_winner(
                            player1_last,
                            player2_last,
                            surface,
                            model_data["player_stats"],
                            model_data["model"],
                            model_data["surface_encoder"]
                        )
                        
                        # Add prediction to match data
                        if prediction_result:
                            match_data["prediction"] = {
                                "winner": prediction_result["prediction"]["winner"],
                                "confidence": prediction_result["prediction"]["confidence"]
                            }
                        else:
                            match_data["prediction"] = {
                                "error": "No prediction available"
                            }
                    except Exception as e:
                        # Don't fail the whole request if one prediction fails
                        print(f"Error predicting match {key}: {e}")
                        match_data["prediction"] = {
                            "error": f"Prediction failed: {str(e)}"
                        }
                
                result["matches"][key] = match_data
        
        return result
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        print(f"Error retrieving tournament matches: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tournament matches: {str(e)}")

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 8004))
    
    # Run the app with uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)