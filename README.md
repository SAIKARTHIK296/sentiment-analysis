<div align="center">

# 🏥 Healthcare Patient Feedback & Sentiment Analytics Engine

**End-to-End Multi-Model Machine Learning Microservice for Patient Sentiment, CSAT, NPS & Ticket Intelligence**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![HTML5](https://img.shields.io/badge/Dashboard-HTML5%2FJS-E34F26?style=for-the-badge&logo=html5&logoColor=white)]()
[![License](https://img.shields.io/github/license/SAIKARTHIK296/sentiment-analysis?style=for-the-badge)](LICENSE)

*Transform unstructured hospital patient feedback into real-time sentiment scores, CSAT/NPS predictions, and automated support ticket intelligence.*

</div>

---

## ⚡ Overview

Healthcare providers process thousands of patient touchpoints daily across departments, discharge surveys, and support channels. Manually triaging this volume leads to missed escalations and delayed service recovery.

This repository delivers a **production-ready ML pipeline and FastAPI microservice** backed by five specialized serialized models (`.pkl`) and an interactive web dashboard (`dashboard2.html`). It automates:
* **Patient Sentiment & Feedback Classification** — Real-time polarity and complaint detection.
* **CSAT & NPS Forecasting** — Predictive scoring for Customer Satisfaction and Net Promoter Score trends.
* **Automated Ticket Intelligence** — Classification and prioritization of hospital support tickets.
* **Unified Hospital ML Inference** — Consolidated scoring through a central inference pipeline (`mlpipeline3.py`).

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[Patient Feedback / Ticket Data] --> B[FastAPI Microservice<br/>fastapimricoservice3.py]
    B --> C[ML Inference Pipeline<br/>mlpipeline3.py]
    
    subgraph Models [Serialized ML Models .pkl]
        M1[best_hospital_ml_model.pkl]
        M2[feedback_best_model.pkl]
        M3[csat_best_model.pkl]
        M4[nps_best_model.pkl]
        M5[ticket_best_model.pkl]
    end
    
    C --> Models
    Models --> C
    C --> D[JSON Prediction Response]
    D --> E[Interactive Analytics Dashboard<br/>dashboard2.html]
