# 🚦 Smart Traffic Management System

A comprehensive MLOps project for traffic prediction and accident risk assessment using machine learning models deployed with MLflow and Docker.

## 📋 Overview

This project implements a real-time traffic management dashboard that:
- **Predicts traffic volume** using XGBoost regression
- **Assesses accident risk** using LightGBM classification
- Provides actionable insights for traffic control authorities
- Deploys models using MLflow Model Registry
- Containerized with Docker for easy deployment

## 🎯 Features

### Traffic Volume Prediction
- Forecasts vehicle traffic volume based on temporal and environmental factors
- Uses XGBoost regression model
- Provides congestion level assessment (Low/Medium/High)
- Estimates average vehicle speed

### Accident Risk Assessment
- Predicts accident probability with risk scoring (0-100%)
- Adjusts risk based on weather conditions, signal status, and time factors
- Provides risk levels: 🟢 LOW, 🟡 MEDIUM, 🔴 HIGH
- Offers actionable recommendations for traffic authorities

### Interactive Dashboard
- Built with Streamlit
- Real-time predictions with adjustable parameters
- Visual metrics and risk factor analysis
- Historical data visualization

## 🏗️ Architecture

```
├── Smart_traffic_prediction.ipynb  # Model training notebook
├── dashboard.py                     # Streamlit dashboard
├── smart_traffic_management_dataset.csv
├── mlruns/                          # MLflow tracking data
├── Dockerfile                       # Container configuration
├── requirements.txt                 # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)
- Git

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd MLOPS
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Training Models

Run the Jupyter notebook to train and register models:

```bash
jupyter notebook Smart_traffic_prediction.ipynb
```

Or execute all cells:
```bash
jupyter nbconvert --to notebook --execute Smart_traffic_prediction.ipynb
```

This will:
- Process the traffic dataset
- Train XGBoost model for traffic volume prediction
- Train LightGBM model for accident risk classification
- Register models in MLflow Model Registry
- Log metrics and artifacts

### Running Locally

```bash
streamlit run dashboard.py --server.port 8501
```

Access the dashboard at `http://localhost:8501`

### Docker Deployment

1. **Build the Docker image**
```bash
docker build -t traffic-dashboard .
```

2. **Run the container**
```bash
docker run --rm -it -p 8501:8501 \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -v "$(pwd)/dashboard.py:/app/dashboard.py" \
  -v "$(pwd)/smart_traffic_management_dataset.csv:/app/smart_traffic_management_dataset.csv" \
  traffic-dashboard:latest
```

Access at `http://localhost:8501`

## 📊 Dataset Features

### Input Features
- **Temporal**: hour, day_of_week, month, is_weekend, is_rush_hour
- **Environmental**: weather_condition (Sunny, Cloudy, Rainy, Foggy, Windy)
- **Infrastructure**: signal_status (Green, Yellow, Red)
- **Traffic Metrics**: vehicle counts (cars, trucks, bikes), avg_vehicle_speed
- **Conditions**: temperature, humidity

### Target Variables
- `traffic_volume`: Number of vehicles (regression)
- `accident_reported`: Binary classification (0/1)

## 🔧 Model Details

### XGBoost Traffic Forecaster
- **Task**: Regression
- **Metrics**: RMSE, MAE
- **Features**: 18 input features
- **Registered Model**: `XGBoost_Forecaster`

### LightGBM Accident Predictor
- **Task**: Binary Classification
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Features**: 18 input features
- **Handles**: Class imbalance with scale_pos_weight
- **Registered Model**: `LightGBM_Accident_Predictor`

### Risk Adjustment Logic
The accident risk assessment uses model probability + adjustment factors:
- **Weather Impact**: +25% for Rainy/Foggy/Windy
- **Signal Status**: +30% for Red, +15% for Yellow
- **Rush Hour**: +15% during peak times (7-9 AM, 5-7 PM)

**Risk Levels**:
- 🟢 LOW: < 30%
- 🟡 MEDIUM: 30-60%
- 🔴 HIGH: > 60%

## 📈 MLflow Integration

View experiment tracking and model registry:

```bash
mlflow ui --port 5000
```

Access at `http://localhost:5000`

## 🛠️ Technologies Used

- **Machine Learning**: XGBoost, LightGBM, scikit-learn
- **MLOps**: MLflow (tracking, registry, model serving)
- **Dashboard**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Containerization**: Docker

## 📝 Usage Examples

### Predict Traffic Volume
1. Select "Traffic Volume" prediction type
2. Adjust sliders: hour, day, month
3. Choose weather and signal status
4. Click "🔮 Predict Traffic Volume"

### Assess Accident Risk
1. Select "Accident Risk" prediction type
2. Configure parameters (e.g., Rainy + Red signal)
3. Click "🔮 Predict Accident Risk"
4. Review risk score, level, and recommendations

## 🔍 Project Structure Details

```
MLOPS/
├── mlruns/                          # MLflow experiment tracking
│   └── 593742697345910866/          # Experiment ID
│       └── <run_ids>/               # Individual run artifacts
├── Smart_traffic_prediction.ipynb   # Training pipeline
├── dashboard.py                     # Streamlit application
├── smart_traffic_management_dataset.csv
├── requirements.txt                 # Python packages
├── Dockerfile                       # Container definition
└── README.md                        # This file
```

## 🐛 Troubleshooting

### Models not loading in Docker
- Ensure volume mounts are correctly specified
- Check MLflow tracking URI points to `/app/mlruns`

### Port already in use
```bash
lsof -i :8501  # Find process
kill <PID>     # Stop it
```

### Package installation errors
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Traffic dataset source
- MLflow documentation
- Streamlit community
- XGBoost and LightGBM teams

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Contact: your.email@example.com

---

**⭐ If you find this project useful, please star the repository!**
