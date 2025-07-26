# mlops-assignment3
End-to-End MLOps Pipeline using Sklearn, Docker, PyTorch, and GitHub Actions


MLOps Assignment 3: End-to-End MLOps Pipeline
Submitted by:
Name: Umang Garg

GitHub Repo: https://github.com/g24ai1102/mlops-assignment3

DockerHub Repo: https://hub.docker.com/u/g24ai1102


Project Structure & Objective:
This project demonstrates a complete MLOps pipeline using:

Linear Regression (scikit-learn)

PyTorch for model conversion

Manual quantization (uint8)

Docker for containerization

GitHub Actions for CI/CD automation


Git Branching Strategy:
| Branch         | Purpose                             |
| -------------- | ----------------------------------- |
| `main`         | Initial setup, base branch          |
| `dev`          | Model training using sklearn        |
| `docker_ci`    | Docker + GitHub Actions CI/CD       |
| `quantization` | Quantized PyTorch model & inference |


Step-by-Step Overview
1. dev Branch - Model Training
Trained a LinearRegression model on the California Housing dataset.

Saved the model using joblib → models/model.joblib

2. docker_ci Branch - Docker + CI/CD
Created Dockerfile and predict.py

Verified container works via GitHub Actions workflow

Pushed Docker image to DockerHub

3. quantization Branch - Manual Model Optimization
Extracted coef_ and intercept_ from sklearn model

Performed manual quantization to uint8

Built a PyTorch model using dequantized weights

Evaluated the R² score and saved parameters using joblib


| Metric              | Original Sklearn Model | Quantized Model |
| ------------------- | ---------------------- | --------------- |
| **R² Score**        | \~0.6                  | -0.1542         |
| **Model Size (KB)** | 4 KB                   | 4 KB            |


Requirements:
pip install -r requirements.txt


Run Locally: For Training
python src/train.py


Predict using Docker:
docker build -t mlops-predict .
docker run mlops-predict


Quantized Inference:
python src/quantize.py


CI/CD Summary
The GitHub Actions pipeline:

Runs on push to docker_ci

Installs dependencies

Trains model (train.py)

Builds and tests Docker image

Pushes image to DockerHub

Workflow file: .github/workflows/ci.yml


Author Note:
This project is a submission for MLOps Assignment 3 under IIT Jodhpur’s coursework.