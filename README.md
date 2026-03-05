
# 🏭 Factory Reallocation & Shipping Optimization System

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn)

**A Decision Intelligence Dashboard built for Nassau Candy Distributor to optimize logistics, reduce shipping lead times, and prevent margin erosion.**

---

## 📌 Project Overview
Nassau Candy currently assigns products to factories using static rules and legacy processes, leading to suboptimal shipping distances, high lead times, and margin erosion. 

This project elevates the supply chain from descriptive analytics to **Decision Intelligence**. It introduces a machine learning-backed recommendation engine that:
1. **Predicts** shipping outcomes under different origin-destination configurations.
2. **Simulates** "What-If" factory reassignment scenarios.
3. **Recommends** optimal product-to-factory configurations that balance shipping speed with profitability.

---

## 📸 Dashboard Preview
![Streamlit Dashboard Placeholder](https://via.placeholder.com/800x400?text=Insert+Your+Streamlit+Screenshot+Here)
*Add a screenshot of your actual Streamlit Dashboard here by uploading an image to your repository and updating this link.*

---

## ⚙️ Core Architecture & Methodology

### 1. Data Processing & Geospatial Analytics
* **Geocoding:** Maps customer postal codes and factory coordinates using latitude/longitude datasets.
* **Distance Calculation:** Computes the true **Haversine distance** (in miles) between distribution centers and final delivery points.
    
    The distance $d$ is calculated using the Haversine formula:
    $$d = 2r \arcsin\left(\sqrt{\sin^2\left(\frac{\phi_2 - \phi_1}{2}\right) + \cos(\phi_1) \cos(\phi_2) \sin^2\left(\frac{\lambda_2 - \lambda_1}{2}\right)}\right)$$
    *Where $r$ is the Earth's radius, $\phi$ is latitude, and $\lambda$ is longitude.*



### 2. Predictive Modeling (The Brain)
* **Algorithm:** `RandomForestRegressor` 
* **Objective:** Predicts the expected **Actual Lead Time** (in days) based on `Distance_Miles`, `Ship Mode`, `Region`, and `Origin_Factory`.
* **Evaluation:** Optimized for low Root Mean Squared Error (RMSE) and high Variance Explained ($R^2$).

### 3. Scenario Simulation Engine (The Logic)
* Intercepts standard routing logic to temporarily "move" a product across all available factories.
* Recalculates distances and feeds them into the trained Random Forest model.
* Ranks factory options based on **Lead Time Reduction**, **Risk Reduction**, and **Profit Impact**.

---

## 📊 Streamlit Application Features
* **Factory Optimization Simulator:** Select a product and instantly view predicted performance across all alternate factories.
* **What-If Scenario Analysis:** Compare current vs. recommended assignments and visualize lead-time improvements.
* **Risk & Impact Panel:** Triggers profit impact alerts and warns leadership of high-risk reassignments.
* **Dynamic Sliders:** Allows stakeholders to adjust the Optimization Priority logic (Speed vs. Profit).

---

## 🚀 How to Run the Application Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/Sahil2171/nassau-candy-logistics-optimizer.git](https://github.com/Sahil2171/nassau-candy-logistics-optimizer.git)
cd nassau-candy-logistics-optimizer

```

**2. Install dependencies:**

```bash
pip install pandas numpy scikit-learn streamlit

```

**3. Train the Machine Learning Model (Optional):**
To process the raw .csv data, calculate distances, and generate a fresh `.pkl` model, run:

```bash
python train_model.py

```

**4. Launch the Web Dashboard:**

```bash
streamlit run app.py

```

The app will automatically open in your browser at `http://localhost:8501`.

---

## 🛠 Tech Stack

| Component | Technology |
| --- | --- |
| **Language** | Python 3.x |
| **Frontend** | Streamlit |
| **Data Handling** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn (Random Forest) |
| **Geospatial** | Geopy / Haversine formula |

---

**Developed by Sahil** | [LinkedIn](https://www.linkedin.com/in/sahilpatil2171/) | [GitHub](https://github.com/Sahil2171)

