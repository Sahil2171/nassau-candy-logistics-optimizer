# 🏭 Factory Reallocation & Shipping Optimization System
**Decision Intelligence Dashboard for Nassau Candy Distributor**

This repository contains an end-to-end Machine Learning Operations (MLOps) pipeline and interactive web dashboard designed to optimize logistics, reduce shipping lead times, and prevent margin erosion.

## 📌 Project Overview
Traditional logistics often rely on static rules for assigning products to distribution centers. This project introduces **Decision Intelligence** to Nassau Candy's supply chain by:
1. **Predicting** shipping outcomes under different configurations using Machine Learning.
2. **Simulating** "What-If" scenarios to test product reassignments.
3. **Recommending** optimal factory-to-product mappings that balance shipping efficiency with profitability.

## ⚙️ Features
* **Predictive Modeling:** Utilizes a `RandomForestRegressor` to accurately forecast shipping lead times based on distance, shipping mode, and region.
* **Geospatial Analytics:** Calculates Haversine distances between manufacturing factories and customer zip codes to evaluate route efficiency.
* **What-If Scenario Simulator:** Allows stakeholders to dynamically test the impact of moving a product line from one factory to another.
* **Risk & Impact Panel:** Automatically flags high-risk reassignments that might improve speed but destroy profit margins.

## 🧰 Tech Stack
* **Language:** Python 3.x
* **Data Processing:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Web Framework:** `streamlit`

## 📊 Key Performance Indicators (KPIs) Targeted
* **Lead Time Reduction (%):** Operational efficiency gain.
* **Profit Impact Stability:** Ensuring faster shipping maintains financial safety.
* **Scenario Confidence Score:** ML reliability metric for recommendations.
* **Recommendation Coverage:** Scalability across the product catalog.

## 🚀 How to Run the Application Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Sahil2171/nassau-candy-logistics-optimizer.git](https://github.com/Sahil2171/nassau-candy-logistics-optimizer.git)
   cd nassau-candy-logistics-optimizer
