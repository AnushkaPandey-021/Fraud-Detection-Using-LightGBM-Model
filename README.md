# Fraud-Detection-Using-LightGBM-Model

### **Project Title:** Financial Fraud Detection Using LightGBM and Streamlit Dashboard

**Why:**
With the increase in digital financial transactions, detecting fraud in real time is critical. Traditional systems often fail to adapt to changing fraud patterns or new data structures. This project addresses that gap by offering a generalizable machine learning solution for fraud detection with visual, interactive analysis.

**What:**
* Developed a fraud detection system using **LightGBM** for efficient, high-accuracy predictions.
* Built a **Streamlit-based dashboard** to allow users to upload any financial CSV file and receive **real-time fraud predictions and insights**.
* Designed to handle unseen datasets with varying features, sizes, and missing data.
* Emphasized visual storytelling through fraud gauges, animated charts, and distribution graphs.

**How:**
* Trained on the **Kaggle PaySim dataset**, with custom preprocessing to adapt to new column structures.
* Engineered features around transaction types, balances, ratios, and timing for better fraud detection.
* Used **chunk-wise loading** for memory-efficient processing of large files.
* Created an intuitive UI with **animated visuals** and **downloadable results** to support business users and fraud analysts.

**Future Scope & Improvements:**
* **Explainability with SHAP (SHapley Additive exPlanations):** To interpret model predictions, understand **why** a transaction was flagged as fraudulent, and identify the most influential features.
* **Integration of XAI (Explainable AI):** For generating **automatic textual explanations** and **risk factor summaries** for each flagged transaction.
* **Auto-alert system:** Email or notification triggers when fraud probability crosses a threshold.
* **Real-time streaming support:** Extending the system to handle **live transaction data** via APIs or Kafka.
* **Feedback loop:** Users can mark false positives/negatives, helping the model improve over time.
* **Model retraining module:** To periodically fine-tune the model using new data and reduce drift.

![image](https://github.com/user-attachments/assets/eab577a2-cb15-421d-8dc9-e1d0ec497b1a)
