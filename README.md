# 🚦 ML-Based Adaptive Traffic Management System

An adaptive traffic signal control system that uses **Machine Learning**, **Priority Queue scheduling**, and **Emergency Vehicle Preemption** to optimize traffic flow at a four-way intersection.  
The system dynamically adjusts signal timing based on real-time traffic conditions instead of fixed-time control.

---

## 📌 Overview

Traditional traffic signals operate on static timers and fail to respond to changing traffic density or emergency situations.  
This project simulates an **intelligent traffic management system** that:

- Predicts lane congestion using Machine Learning
- Dynamically schedules green signals using a Priority Queue
- Instantly prioritizes emergency vehicles
- Visualizes traffic behavior in real time

The project is implemented using **Python**, with a **Pygame-based simulation** and a **Flask dashboard** for live analytics.

---

## 🎯 Key Features

- Machine Learning–based congestion prediction
- Priority Queue–driven adaptive signal control
- Emergency Vehicle Preemption (ambulance priority)
- Fair signal allocation with starvation avoidance
- Real-time traffic simulation
- Live dashboard for monitoring traffic metrics

---

## 🧠 System Workflow

1. Traffic data (vehicle count, waiting time, queue length, emergency presence) is generated in the simulation.
2. A trained ML model predicts congestion levels (Low / Medium / High) for each lane.
3. A **priority score** is computed for every lane using:
   - Vehicle count
   - Waiting time
   - Predicted congestion
   - Emergency presence
4. A **Priority Queue** selects the lane with the highest urgency for the next green signal.
5. If an emergency vehicle is detected, normal operation is overridden until clearance.
6. Traffic states and metrics are displayed on a live dashboard.

---

## 🤖 Machine Learning Model

- **Algorithm:** Random Forest Classifier
- **Purpose:** Predict congestion level per lane
- **Input:** Traffic density, queue metrics, time features, turn complexity
- **Output:** Congestion category (Low / Medium / High)

### Model Performance
- Accuracy: ~95%
- Balanced precision, recall, and F1-score
- Reliable congestion classification for real-time control

---

## ⏱️ Priority Queue Scheduling

Each lane is treated as an independent entity in a **priority queue**.

The priority score dynamically combines:
- Number of waiting vehicles
- Average waiting time
- Predicted congestion level
- Emergency flag
- Starvation compensation

The lane with the highest score receives the green signal, ensuring both **efficiency and fairness**.

---

## 🚑 Emergency Vehicle Preemption

- Detects ambulances within the simulation
- Overrides normal signal scheduling
- Grants immediate green signal to the emergency lane
- Automatically restores adaptive control after clearance

This ensures minimal delay for emergency vehicles while maintaining intersection stability.

---

## 🎮 Simulation Environment

- **Framework:** Pygame
- **Intersection:** 4-way junction
- **Lanes:** Left, Straight, Right per direction
- **Vehicle Types:** Cars, buses, trucks, bikes, rickshaws, ambulances
- **Signal Timing:** Dynamically adjusted (10–60 seconds)

---

## 📊 Real-Time Dashboard

Built using **Flask and Chartjs**, the dashboard displays:
- Current signal states
- Congestion levels per lane
- Vehicle counts and waiting times
- Throughput and delay metrics
- Emergency mode indicators

This improves transparency and system interpretability.

---

## 📂 Project Structure
```bash
Traffic-simulation/
│
├── traffic_sim.py # Main simulation logic
├── model_training.py # ML model training
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitignore # Ignored files
└── data/ # Datasets & models (not included)
```

---

## ⚠️ Dataset & Models

Due to GitHub file size limits, **datasets and trained ML models are not included** in this repository.

To run the project:
1. Place the dataset inside a `data/` directory
2. Run the training script to generate models locally

This follows GitHub and ML best practices.

---

## 🚀 How to Run

```bash
git clone https://github.com/ShouryaDesai135/Traffic-simulation.git
cd Traffic-simulation
pip install -r requirements.txt
python traffic_sim.py
```

---

## Authors

Vardhan M. Dhavale,
Shourya S. Desai,
Saumya S. Dhorje,
Ananya A. Gaikwad,
Payal S. Dhaygude,

Guided by Sheela V. Chinchmalatpure

Vishwakarma Institute of Technology, Pune
