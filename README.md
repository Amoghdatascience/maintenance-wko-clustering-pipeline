# Aviation Maintenance Workorder Clustering Pipeline

## Overview
This project focuses on clustering aviation maintenance workorders (WKOs) to transform unstructured maintenance text into structured categories. These clusters serve as a foundation for predictive maintenance (PdM) and anomaly detection in aviation systems.

Maintenance workorders contain noisy, domain-specific text with abbreviations and inconsistent terminology, making direct analysis difficult. This pipeline uses a hybrid approach combining rule-based logic, semantic similarity, and machine learning to improve clustering accuracy and scalability.

---

## Key Features
- Hybrid clustering approach (Rules + ML + Semantic Similarity)
- Hierarchical clustering (Parent :- Child clusters)
- Aviation-specific text normalization and abbreviation expansion
- TF-IDF with word + character n-grams
- Logistic Regression for classification
- Cluster-wise dataset generation
- Designed for downstream predictive maintenance applications

---

## Pipeline Architecture

The clustering pipeline follows a multi-stage approach:

1. Text Preprocessing  
   - Combine PROBLEM and ACTION  
   - Normalize text (lowercase, clean noise)  
   - Expand aviation-specific abbreviations  

2. Rule-Based Clustering  
   - Hard rules (strict domain mapping)  
   - Seed rules (initial labeling)  
   - Split into Known and Unknown data  

3. Parent Clustering  
   - Assign unknown records to system-level clusters  
   - Uses TF-IDF + Logistic Regression + Cosine Similarity  

4. Child Clustering  
   - Assign fine-grained clusters  
   - Uses stricter validation thresholds  

5. Semantic Similarity  
   - Matches workorders to cluster descriptions  
   - Acts as fallback  

6. Hard Rule Overrides  
   - Final domain-based corrections  

---

## Project Structure

---

## How to Run the Project

### 1. Prerequisites

- Python 3.8+
- pip
- black

Install dependencies:
```
pip install pandas numpy scikit-learn scipy openpyxl
```
---

### 2. Required Input Files

Place these files in the root directory:

- Valid_Problems_Workorder.xlsx  
- valid-WKO_and_component_times.csv  

---

### 3. Step-by-Step Execution

Step 1: Run Clustering Pipeline
```
python semantic_clustering_pipe.py
```
This will:
- Normalize aviation text
- Apply rule-based + ML clustering
- Generate dataset.csv and summary.csv

---

Step 2: Merge and Generate Cluster Outputs
```
python merge.py
```
This will:
- Join datasets
- Aggregate workorders
- Generate cluster_outputs/


---
Step 3: Run test cases
```
python -m pytest -v tests/test_pipeline.py
```
The test cases validate important parts of the pipeline, including:
- text normalization
- abbreviation expansion
- strict baffle token rules
- flight control overrides
- dirty/filter/tire override logic
- final dataset join behavior
---
### 4. Final Outputs

- dataset.csv :- Clustered workorders  
- summary.csv :- Clustering metrics  
- cluster_outputs/ :- Cluster-wise CSV files  

---

### 5. Important Notes

- Always run semantic_clustering_pipe.py before merge.py  
- Ensure correct column names (WKO#, PROBLEM, ACTION, etc.)  
- Large datasets may take time  

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy

---

## Design Decisions

- Hybrid approach balances precision and generalization  
- Hierarchical clustering reduces errors  
- TF-IDF captures semantics and abbreviations  
- Logistic Regression works well on sparse data  

---

## Future Work

- Flight time-series integration  
- Flight phase segmentation  
- Anomaly detection  
- Predictive maintenance modeling  

---

## Author

Amogh Naik  
MS Data Science, Rochester Institute of Technology  

Advisor: Dr. Travis Desell  


