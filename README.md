# Predictive Maintenance & IoT Forecasting Engine ⚙️📈

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Regression-success.svg)
![Domain](https://img.shields.io/badge/Domain-IoT%20%26%20Edge%20Hardware-purple.svg)

## Project Overview
A robust deep learning pipeline designed to forecast the **Remaining Useful Life (RUL)** of enterprise edge-devices (such as Wi-Fi routers and IoT gateways). By analyzing continuous time-series telemetry data (CPU temperature, memory utilization, packet drop rates), this engine predicts impending hardware failures before they occur, enabling data-driven, preemptive maintenance protocols.

## Key Highlights
* **High-Accuracy Forecasting:** Forecasted the Remaining Useful Life (RUL) of enterprise hardware systems with a Mean Absolute Error (MAE) of **27.67 hours** by deploying a deep feedforward neural network.
* **Stochastic Data Modeling:** Simulated complex degradation patterns across a 500-hour lifecycle for 100 edge devices by generating synthetic sensor telemetry data infused with environmental noise.
* **Advanced Feature Engineering:** Enhanced the model's temporal awareness, improving predictive stability, by engineering rolling time-domain features and discrete change-rate indicators.

## Tech Stack & Applied Mathematics
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras (Sequential API)
* **Data Processing:** Pandas, NumPy, Scikit-learn (MinMaxScaler)
* **Mathematical Core:** * **Statistics:** Gaussian noise $N(\mu=0, \sigma^2)$ for stochastic modeling.
  * **Calculus:** Discrete derivatives ($\frac{\Delta y}{\Delta x}$) to calculate the rate of thermal degradation, and Gradient Descent via the Adam optimizer.

## System Architecture

### 1. Stochastic Data Synthesis
Instead of relying on static datasets, this project generates a realistic, highly noisy IoT telemetry dataset from scratch. It simulates 100 enterprise routers over a 500-hour maximum lifespan. The degradation of hardware (temperature spikes, memory leaks) is modeled using exponential curves combined with randomized Gaussian noise to simulate real-world environmental unpredictability.

### 2. Temporal Feature Engineering
Neural networks require context to understand *how fast* something is breaking.
* **Rolling Means:** Applied to smooth out extreme Gaussian noise spikes.
* **Discrete Derivatives:** Calculates the hour-over-hour rate of change ($\frac{\Delta y}{\Delta x}$) for CPU temperature, giving the AI explicit awareness of accelerating hardware failure.

### 3. Deep Feedforward Neural Network
The regression model is built to handle non-linear degradation curves.
* **Data Splitting:** Data is strictly split by `router_id` (Train on routers 1-80, Test on routers 81-100) to completely prevent time-series data leakage.
* **Architecture:** Dense layers (64 -> 32 -> 16 -> 1) with ReLU activations. 
* **Regularization:** Dropout layers (20%) are implemented to prevent the network from memorizing the statistical noise.
* **Optimizer:** Adam (Adaptive Moment Estimation) minimizing Mean Squared Error (MSE).

### 4. Evaluation Metrics
The model is evaluated on entirely unseen edge devices. Performance is measured using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** to quantify exactly how many hours off the AI's prediction is compared to the actual time of hardware death.

## Results & Visualizations
The model successfully locked onto the exponential degradation curve of unseen test routers, achieving a final **MAE of 27.67 hours**. Given a 500-hour total lifecycle (~20 days), predicting hardware death within a roughly 24-hour margin of error allows for highly effective preemptive maintenance scheduling.
