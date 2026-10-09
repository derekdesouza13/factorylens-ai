# FactoryLens AI

### Manufacturing Quality, Predictive Maintenance & GenAI Decision Intelligence Platform

FactoryLens AI is an end-to-end manufacturing analytics platform that combines
**data engineering, predictive machine learning, explainable AI, MLOps, and
Generative AI** to identify production quality risks, detect machine anomalies,
and provide actionable operational insights.

---

## Overview

Manufacturing environments generate data across production, machine sensors,
quality inspections, energy consumption, and maintenance systems.

FactoryLens AI consolidates these heterogeneous datasets into a unified
analytics pipeline to answer key operational questions:

- Which production batches are at risk of quality issues?
- Which machines are showing abnormal behavior?
- What factors are driving a model's prediction?
- Which machines require maintenance investigation?
- Can engineers query production insights using natural language?

---

## Key Features

- **Multi-source Data Integration** — Production, machine, quality, energy,
  and maintenance datasets.
- **Automated Data Preparation** — Cleaning, validation, aggregation,
  feature engineering, and preprocessing.
- **Predictive Quality Analytics** — Classification models for production
  quality-risk prediction.
- **Machine Anomaly Detection** — Identification of abnormal machine behavior
  and operational risk.
- **Explainable AI** — SHAP-based explanations for model predictions.
- **ML Experiment Tracking** — Model experiments, metrics, artifacts, and
  model versions using MLflow.
- **Analytics Dashboard** — Production, quality, machine, energy, and model
  performance monitoring.
- **GenAI Manufacturing Copilot** — Natural-language interaction with
  manufacturing data and ML insights.
- **RAG & AI Agents** — LangChain/LangGraph-based retrieval and tool-calling
  workflows for operational analysis.
- **Production-ready ML Workflow** — FastAPI, Docker, automated testing,
  and CI/CD.

---

## Architecture

```text
Production ─────┐
Machine Sensors ─┤
Quality Data ────┤
Energy Data ─────┤
Maintenance ─────┘
        │
        ▼
┌──────────────────────┐
│ Data Ingestion       │
│ Cleaning & Validation│
│ Aggregation          │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Feature Engineering  │
│ EDA & Data Analysis  │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Machine Learning     │
│ Quality Prediction   │
│ Anomaly Detection    │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ SHAP Explainability  │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ MLflow / Model       │
│ Tracking & Registry  │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ FastAPI Model Serving│
└──────────┬───────────┘
           │
      ┌────┴─────┐
      ▼          ▼
 Dashboard    GenAI Copilot
              │
       ┌──────┴──────┐
       ▼             ▼
      RAG        LangGraph
                    Agents
