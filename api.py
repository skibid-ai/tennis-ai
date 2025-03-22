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
from typing import List, Dict, Optional, Any

# Import functions from main.py
from main import load_data, extract_features, train_model, predict_winner

app = FastAPI(
    title="Tennis Match Predictor API",
    description="API to predict tennis match outcomes based on historical data",
    version="1.0.0"
)

# Global variables to store model and data
model = None
player_stats = None
surface_encoder = None
model_ready = False
last_load_time = 0
MODEL_RELOAD_INTERVAL = 3600  # Reload model every hour

# Define request and response models
class MatchPredictionRequest(BaseModel):
    playerOne: str = Field(..., description="Last name of the first player")
    playerTwo: str = Field(..., description="Last name of the second player")
    surface: str = Field("Hard", description="Court surface (e.g., 'Hard', 'Clay', 'Grass')")

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

def get_model():
    """Dependency to ensure model is loaded."""
    if not model_ready:
        initialize_model()
        if not model_ready:
            raise HTTPException(status_code=500, detail="Model initialization failed")
    return {"model": model, "player_stats": player_stats, "surface_encoder": surface_encoder}

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    initialize_model()

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", model_ready=model_ready)

@app.post("/predict", response_model=MatchPredictionResponse, tags=["Prediction"])
async def predict(data: MatchPredictionRequest, model_data: Dict = Depends(get_model)):
    """Predict the winner of a tennis match."""
    try:
        # Make prediction
        result = predict_winner(
            data.playerOne,
            data.playerTwo,
            data.surface,
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

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 8000))
    
    # Run the app with uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False) 