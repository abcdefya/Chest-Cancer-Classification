# Chest Cancer Classification

## Project Overview
This project implements a deep learning solution for classifying chest CT scans to detect adenocarcinoma cancer. It uses a VGG16-based convolutional neural network to analyze medical images and classify them as either normal or showing signs of adenocarcinoma cancer.

## Tech Stack
- Python 3.11
- TensorFlow 2.12.0
- Flask
- DVC (Data Version Control)
- MLflow
- Docker

## Project Structure
The project follows a modular architecture with the following components:

- **Data Ingestion**: Downloads and extracts the dataset
- **Base Model Preparation**: Configures the VGG16 model for transfer learning
- **Model Training**: Trains the model with data augmentation
- **Model Evaluation**: Evaluates model performance and logs metrics

## Features
- End-to-end ML pipeline with DVC for reproducibility
- Web interface for real-time predictions
- Docker containerization for easy deployment
- CI/CD pipeline using GitHub Actions
- AWS integration for model deployment

## Installation & Setup

### Prerequisites
- Python 3.11
- Docker (optional)

### Local Setup
1. Clone the repository
```bash
git clone https://github.com/your-username/Chest-Cancer-Classification.git
cd Chest-Cancer-Classification
```

2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Running the Application

#### Using Python
```bash
python app.py
```

#### Using Docker
```bash
docker build -t chest-cancer-classifier .
docker run -p 8080:8080 chest-cancer-classifier
```

## Model Training Pipeline

To run the complete training pipeline:
```bash
python main.py
```

Or using DVC:
```bash
dvc repro
```

## API Endpoints

- **GET /** - Web interface for uploading and classifying images
- **POST /predict** - API endpoint for image classification
- **GET /train** - Trigger model retraining

## Project Configuration

Configuration parameters are stored in:
- `params.yaml`: Model hyperparameters
- `config/config.yaml`: Pipeline configuration

## Deployment

The project includes a complete CI/CD workflow for AWS deployment:
1. Code is tested and linted
2. Docker image is built and pushed to Amazon ECR
3. Application is deployed to a self-hosted runner

This guy help me to land a AI Engineer job
