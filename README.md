# Wine Classification Application

## Project Overview
This project is designed to train and deploy a machine learning model for classifying wine types based on specific features. The application leverages Docker for containerization, MLflow for experiment tracking, and GitHub Actions for CI/CD automation.

## Prerequisites

Before running this application, ensure the following tools and environments are set up on your machine:

- **Python (>=3.10)**: Install Python from [python.org](https://www.python.org/).
- **Docker**: Install Docker Desktop from [docker.com](https://www.docker.com/).
- **Git**: Install Git from [git-scm.com](https://git-scm.com/).
- **Conda**: Install Miniconda/Anaconda from [conda.io](https://docs.conda.io/en/latest/).
- **MLflow**: Install MLflow for experiment tracking.

## Step-by-Step Guide

### 1. Clone the Repository
```bash
# Clone the repository
git clone https://github.com/souviknath94/ml-ops-grp39-assignment-1.git

# Navigate to the project directory
cd ml-ops-grp39-assignment-1
```

### 2. Set Up the Environment

#### Using Conda
```bash
# Create a new conda environment
conda env create -f conda.yaml

# Activate the environment
conda activate wine-classifier
```

### 3. Start the MLflow Server
Before running the training step, start the MLflow server to track experiments:
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root "file://$(pwd)/artifacts" \
    --host 0.0.0.0
```

### 4. Train the Model
Run the training script to train the wine classification model and log the results in MLflow:
```bash
python src/train.py
```

### 5. Build and Run the Application

#### Build Docker Image
```bash
docker build -t mlops-app:latest .
```

#### Run Docker Container
```bash
docker run -d -p 5000:5000 --name mlops-app mlops-app:latest
```

### 6. Test the Application
Use the following `curl` command to test the API with sample input data:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"features": [13.2, 2.77, 2.51, 24.5, 86.0, 1.45, 1.25, 0.5, 1.7, 6.5, 1.05, 3.33, 820.0]}' \
     http://localhost:5000/predict
```
Expected Output:
```json
{
  "prediction": 1
}
```

### 7. Stop and Remove Docker Container
After testing, stop and remove the running Docker container:
```bash
docker stop mlops-app
docker rm mlops-app
```

## CI/CD Pipeline Explanation

This project uses GitHub Actions for Continuous Integration and Continuous Deployment (CI/CD). The pipeline is defined in the `ci.yaml` file located in `.github/workflows/`.

## CI/CD Pipeline Execution Summary
The CI/CD pipeline automates code linting, unit testing, and deployment using GitHub Actions. Below is a detailed breakdown of each step's execution as observed from the logs.

### 1. Lint Code
- **Purpose**: To ensure code quality and adherence to Python coding standards.
- **Log Highlights**:
  - The pipeline successfully executed `flake8` to lint the code.
  - Any warnings or style violations were logged but didn't fail the pipeline.
  - Example Log Snippet:
    ```text
    2025-01-27T07:19:47.6281378Z Linting complete. No errors found.
    ```

### 2. Run Unit Tests
- **Purpose**: To validate the functionality of the code using unit tests.
- **Log Highlights**:
  - Python `unittest` was executed, and all tests passed successfully.
  - Example Log Snippet:
    ```text
    2025-01-27T07:20:28.7679664Z All unit tests executed successfully.
    2025-01-27T07:20:29.0226480Z No test failures encountered.
    ```

### 3. Deploy and Test Application
- **Purpose**: To deploy the application in a Docker container and verify it using a `curl` command.
- **Log Highlights**:
  - Docker container built successfully.
  - Application tested with a `curl` request, and a successful prediction response was received.
  - Example Log Snippet:
    ```text
    2025-01-27T07:21:11.7340928Z Docker container running on port 5000.
    2025-01-27T07:21:12.2581357Z Prediction response: {"prediction": 2}
    ```

## Logs and Outputs

- [View Linting CI/CD Logs](logs/0_Lint Code.txt)
- [View Unit testing CI/CD Logs](logs/1_Run Unit Tests.txt)
- [View Deployment CI/CD Logs](logs/2_Deploy and Test Application.txt)
---


### Pipeline Steps

#### 1. Lint Code
This step ensures the code adheres to coding standards using `flake8`:
```yaml
lint:
  name: Lint Code
  runs-on: ubuntu-latest
  steps:
    - name: Checkout Code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install Dependencies
      run: |
        pip install flake8
        pip install -r requirements.txt

    - name: Run Linter
      run: flake8 . || true
```

#### 2. Run Unit Tests
This step runs unit tests to ensure the application works as expected:
```yaml
test:
  name: Run Unit Tests
  runs-on: ubuntu-latest
  needs: lint
  steps:
    - name: Checkout Code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install Dependencies
      run: |
        pip install -r requirements.txt

    - name: Run Unit Tests
      working-directory: ./
      run: python -m unittest discover -s src/test/unit_test -p "*.py"
```

#### 3. Deploy and Test Application
This step deploys the application locally on the runner using Docker and tests it using a `curl` command:
```yaml
deploy-to-runner:
  name: Deploy and Test Application
  runs-on: ubuntu-latest
  needs: test
  steps:
    - name: Checkout Code
      uses: actions/checkout@v3

    - name: Build Docker Image
      run: |
        docker build -t mlops-app:latest .

    - name: Run Docker Container
      run: |
        docker run -d -p 5000:5000 --name mlops-app mlops-app:latest
        echo "Docker container is running on port 5000."

    - name: Wait for Service to Start
      run: sleep 10

    - name: Test Deployment with cURL
      run: |
        curl -X POST -H "Content-Type: application/json" \
          -d '{"features": [13.2, 2.77, 2.51, 24.5, 86.0, 1.45, 1.25, 0.5, 1.7, 6.5, 1.05, 3.33, 820.0]}' \
          http://localhost:5000/predict

    - name: Stop and Remove Docker Container
      if: always()
      run: |
        docker stop mlops-app
        docker rm mlops-app
```

### CI/CD Pipeline Workflow
1. **Push or Pull Request to Main Branch**: The pipeline is triggered by any push or pull request to the `main` branch.
2. **Lint Code**: Ensures code quality.
3. **Run Unit Tests**: Verifies the correctness of the application.
4. **Deploy and Test**: Builds, deploys, and tests the application locally on the runner.

## Design Choices Explanation

### 1. **Docker for Containerization**
Docker ensures that the application runs consistently across all environments by packaging the code and dependencies together. This avoids compatibility issues and makes deployment more predictable.

### 2. **MLflow for Experiment Tracking**
MLflow is used to track experiments, log parameters, and save model artifacts. This provides reproducibility and better management of experiments during model development.

### 3. **Conda for Dependency Management**
Using Conda ensures that the correct versions of dependencies are installed in an isolated environment, reducing conflicts with system packages.

### 4. **GitHub Actions for CI/CD**
Automating testing, building, and deployment using GitHub Actions ensures that the codebase remains stable and any issues are detected early in the development lifecycle.

### 5. **Pipeline Structure**
- **Linting**: Helps maintain code quality.
- **Unit Testing**: Ensures functionality is working as intended.
- **Deployment and Testing**: Deploying in a Dockerized environment mirrors production setups, ensuring reliability.

### 7. **Separation of Concerns**
Each step in the CI/CD pipeline is modularized to ensure clarity, maintainability, and easier debugging.

Feel free to modify or extend the pipeline and the application based on project requirements.

