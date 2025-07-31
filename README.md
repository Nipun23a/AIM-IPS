# AIM-IPS: Adaptive Intelligent Machine Learning-based Intrusion Prevention System

AIM-IPS is a modular, adaptive security system that leverages anomaly detection, threat classification, and explainable AI to proactively detect and respond to intrusions in web applications. The system integrates honeypots, machine learning, and automated responses to build a comprehensive intrusion prevention pipeline.

## 🔍 Project Structure

aim-ips/
├── honeypot/ # Fake services to collect traffic
├── data_collector/ # Collect, filter and store raw data
├── preprocessing/ # Cleaning, normalization, feature extraction
├── anomaly_detector/ # DL/ML model to flag anomalies
├── classifier/ # Supervised model for threat types
├── explainability/ # SHAP/LIME analysis
├── response_engine/ # Custom actions like IP blocking
├── api/ # FastAPI endpoints
├── dashboard/ (optional) # Admin dashboard UI
├── utils/ # Shared helpers
├── models/ # Trained model files
└── README.md


## 📦 Tech Stack

- **Backend**: Python (FastAPI)
- **ML Frameworks**: Scikit-learn, PyTorch /TensorFlow
- **Explainability**: SHAP, LIME
- **Storage**: MongoDB / PostgreSQL
- **Honeypot**: Cowrie, Dionaea, or Flask-based trap
- **Containerization**: Docker
- **Frontend (Optional)**: React.js or Bootstrap

## 🚀 Modules

Each module in the system is independent and communicates through well-defined APIs or shared data formats. Refer to each subdirectory’s `README.md` for implementation details.

## 🛠️ Getting Started

1. Clone the repo.
2. Install dependencies using `requirements.txt`
3. Set up your honeypot and backend services
4. Run the system using Docker Compose (WIP)


## 📌 License

MIT License

---

