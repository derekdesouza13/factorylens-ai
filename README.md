FactoryLens AI
Manufacturing Quality, Predictive Maintenance & GenAI Decision Intelligence Platform
FactoryLens AI is an end-to-end manufacturing analytics platform that combines machine learning, explainable AI, MLOps, and Generative AI to identify production quality risks, detect machine anomalies, and provide actionable operational insights.
Key Features
- Manufacturing Data Analytics — Integrates production, machine, quality, energy, and maintenance data.
- Predictive ML — Quality-risk prediction and machine anomaly detection using Scikit-learn and XGBoost.
- Explainable AI — SHAP-based feature importance and prediction explanations.
- MLOps — MLflow experiment tracking, model versioning, Docker, and CI/CD.
- Analytics Dashboard — Interactive production, machine health, quality, and model-performance visualizations.
- GenAI Copilot — Natural-language interaction with manufacturing data and ML insights.
- RAG Pipeline — Retrieves relevant information from maintenance and operational documentation.
- AI Agent — LangGraph-based agent with tool calling for querying metrics, predictions, and documentation.
Architecture
Manufacturing Data
       │
       ▼
Data Ingestion & Processing
       │
       ▼
Feature Engineering
       │
       ├───────────────┐
       ▼               ▼
Quality Prediction   Anomaly Detection
       │               │
       └───────┬───────┘
               ▼
        SHAP Explainability
               │
               ▼
            MLflow
               │
               ▼
            FastAPI
               │
       ┌───────┴────────┐
       ▼                ▼
  Analytics        GenAI Copilot
  Dashboard          │
                     ▼
              RAG + LangGraph

Tech Stack
Category	Technologies
Data Science	Python, Pandas, NumPy, Scikit-learn
ML	XGBoost, Random Forest, SHAP
GenAI	LangChain, LangGraph, RAG, LLMs
Backend	FastAPI, PostgreSQL
Visualization	Streamlit, Plotly
MLOps	MLflow, Docker, GitHub Actions
Cloud	AWS


Project Structure
factorylens-ai/
├── data/
├── notebooks/
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── explainability/
├── ml/
├── api/
├── dashboard/
├── rag/
├── agent/
├── tests/
├── docs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

ML Workflow
Data Collection
      ↓
Data Cleaning & Validation
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
SHAP Explainability
      ↓
MLflow Tracking
      ↓
API Deployment

Models are evaluated using Precision, Recall, F1-Score, ROC-AUC and PR-AUC, with particular consideration for class imbalance and model limitations.
GenAI Workflow
The GenAI copilot combines RAG and tool-calling agents to answer questions such as:
Which machines currently have the highest risk?

Why was Machine M14 classified as high risk?

What factors are contributing to the predicted defect?

What maintenance procedure applies to this anomaly?

Getting Started
git clone https://github.com/<username>/factorylens-ai.git
cd factorylens-ai

pip install -r requirements.txt

docker compose up --build

Project Status
In Development
Planned milestones:
- [x] Project architecture
- [ ] Data ingestion & preprocessing
- [ ] Exploratory data analysis
- [ ] Predictive ML models
- [ ] SHAP explainability
- [ ] MLflow integration
- [ ] FastAPI model serving
- [ ] Analytics dashboard
- [ ] RAG pipeline
- [ ] LangGraph agent
- [ ] Docker & CI/CD
Disclaimer
FactoryLens AI is an independent educational and portfolio project using public and/or synthetic manufacturing data. It is not affiliated with or endorsed by Michelin.
Author
Derek Dsouza
B.Tech Computer Science Engineering — MIT World Peace University
GitHub | LinkedIn
