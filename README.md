# 🎾 Tennis Match Predictor

An AI-powered API that predicts the winners of tennis matches based on historical ATP tour data spanning over two decades (2000-2025).

## 🧠 What It Does

This application analyzes over 60,000 professional tennis matches to predict match outcomes using a combination of machine learning and statistical analysis. The system:

- Predicts the most likely winner between any two tennis players
- Provides confidence scores for predictions
- Factors in court surface (Hard, Clay, Grass, Carpet) preferences
- Considers historical rankings and head-to-head performance
- Adapts to player form and surface specialization

## 🔍 Key Features

- **Accurate Predictions**: Uses Random Forest classification enhanced with tennis-specific domain knowledge
- **Surface Intelligence**: Recognizes that players perform differently on various court surfaces
- **Rich Player Data**: Provides detailed statistics for each player in the database
- **Fast API**: Modern FastAPI implementation with auto-documentation
- **Easy Integration**: Simple JSON input/output for integration with websites or applications

## 🛠️ Technical Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- fastapi
- uvicorn
- pydantic

## 📦 Installation

### Option 1: Direct Installation

1. Clone this repository
2. Install the dependencies:
```
pip install -r requirements.txt
```

### Option 2: Docker Installation

Requirements:
- Docker
- Docker Compose (optional)

1. Clone this repository
2. Build and run the Docker container:

```bash
# Using Docker directly
docker build -t tennis-predictor .
docker run -p 8004:8004 tennis-predictor

# OR using Docker Compose (recommended)
docker-compose up
```

## 🚀 Usage

The predictor can be used either as a command-line tool or as a REST API.

### Command Line Usage

```bash
python main.py '{"playerOne": "Federer", "playerTwo": "Nadal", "surface": "Clay"}'
```

### Piping JSON

```bash
echo '{"playerOne": "Federer", "playerTwo": "Nadal", "surface": "Clay"}' | python main.py
```

### REST API

#### Starting the Server

```bash
# Option 1: Direct Python
python api.py

# Option 2: Using Docker
docker-compose up
```

The server will be available at http://localhost:8004 by default.

#### API Documentation

FastAPI automatically generates interactive documentation for the API:
- Swagger UI: http://localhost:8004/docs
- ReDoc: http://localhost:8004/redoc

#### API Endpoints

1. **Predict Match Winner**

   `POST /predict`

   Request body:
   ```json
   {
     "playerOne": "Federer",
     "playerTwo": "Nadal",
     "surface": "Clay"
   }
   ```

   The `surface` field is optional and can be null. If not provided or null, "Hard" surface will be used as default.
   
   Example with null surface (will use "Hard" as default):
   ```json
   {
     "playerOne": "Federer",
     "playerTwo": "Nadal",
     "surface": null
   }
   ```

   Response:
   ```json
   {
     "prediction": {
       "winner": "Nadal",
       "confidence": 0.85,
       "player1_stats": {
         "win_rate": 0.78,
         "surface_win_rate": 0.65,
         "avg_rank": 3,
         "matches_played": 1020
       },
       "player2_stats": {
         "win_rate": 0.81,
         "surface_win_rate": 0.92,
         "avg_rank": 2,
         "matches_played": 980
       }
     }
   }
   ```

2. **Get Available Players**

   `GET /players?search=fed&limit=10`

   Parameters:
   - `search` (optional): Search term to filter players by name
   - `limit` (optional): Maximum number of results to return (default: 50, max: 100)

   Response:
   ```json
   {
     "players": [
       {
         "name": "Federer",
         "avg_rank": 3,
         "win_rate": 0.78,
         "matches_played": 1020
       },
       ...
     ]
   }
   ```

3. **Get Available Court Surfaces**

   `GET /surfaces`

   Response:
   ```json
   {
     "surfaces": ["Hard", "Clay", "Grass", "Carpet"]
   }
   ```

4. **Health Check**

   `GET /health`

   Response:
   ```json
   {
     "status": "healthy",
     "model_ready": true
   }
   ```

## 🧪 Testing

The project includes a test script to verify API functionality:

```bash
python test_api.py
```

This will start the API if not running, test all endpoints, and provide a summary report.

## 📊 How It Works

The prediction model uses a combination of:

1. **Data Preparation**:
   - Historical match results from 2000-2025
   - Player rankings and performance metrics
   - Surface-specific statistics

2. **Feature Engineering**:
   - Win rates overall and by surface
   - Ranking differences between players
   - Match count (experience factor)

3. **Predictive Modeling**:
   - Random Forest classification (70% weight)
   - Surface-specific performance metrics (30% weight)
   - Confidence scoring

## 🏆 Use Cases

- **Sports Betting**: Make informed decisions based on statistical analysis
- **Tennis Fans**: Settle debates about theoretical matchups
- **Tournament Planning**: Predict outcomes for drawing brackets
- **Player Analysis**: Understand strengths and weaknesses on different surfaces
- **Sports Media**: Generate talking points for match previews

## 🐳 Docker Deployment

This project includes a Dockerfile and docker-compose.yml for easy deployment:

### Building and Running with Docker

```bash
# Build the Docker image
docker build -t tennis-predictor .

# Run the container
docker run -p 8004:8004 tennis-predictor
```

### Using Docker Compose

```bash
# Start the API
docker-compose up

# Run in detached mode
docker-compose up -d

# Stop the API
docker-compose down
```

### Accessing the API

Once running in Docker, the API will be available at:
- http://localhost:8004/docs (Swagger UI)
- http://localhost:8004/redoc (ReDoc documentation)

## ⚠️ Limitations

- Predictions are only as good as the historical data
- New players with limited match history will have less accurate predictions
- The model cannot account for current injuries or recent form changes
- Players not found in the database will use default statistics 