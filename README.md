# Production Line Dashboard

An interactive dashboard for analyzing downtime and defect drivers on a copper wire production line, powered by a November 2020 Kaggle dataset.

##Features

- **Data Overview**: Quick metrics on total batches, overall defect rate, and average downtime  
- **Machine-Level Analysis**: Compare all 17 machines—average defect rates and downtime outliers  
- **Shift & Operator Analysis**: Drill into performance across two shifts and 32 operators  
- **Recommendations**: Top parameter combinations ranked by predicted defect probability

##Dataset

The repository includes `dataset.csv`, containing 149 production batches with these columns:

- `Machine` (1–17)  
- `Shift` (A or B)  
- `Operator` (ID 1–32)  
- `Date`  
- `Cable Failures` & `Cable Failure Downtime`  
- `Other Failures` & `Other Failure Downtime`

##Install Dependencies
pip install -r requirements.txt

##Usage
streamlit run A2.py
