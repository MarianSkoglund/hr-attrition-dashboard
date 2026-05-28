# HR Attrition Management Dashboard

A polished Streamlit dashboard for executive HR attrition analysis. It uses the local HR demo dataset when available and falls back to deterministic demo data if no dataset file is present.

## Dataset

Place one of these files in this folder:

- `hr_attrition_demo_data.csv`
- `hr_attrition_demo_data.xlsx`

The app also searches for nearby HR or attrition CSV/XLSX files in the same directory. Column names are normalized so common variants like `Monthly Income`, `JobRole`, `BusinessTravel`, and `EnvironmentSatisfaction` still work.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Deploy To Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Choose **New app**.
4. Select the repository, branch, and `app.py` as the main file.
5. Deploy the app.

The app will use `hr_attrition_demo_data.csv` or `hr_attrition_demo_data.xlsx` if either file is included in the repository. If no dataset is present, it opens with generated demo data.

## Dashboard Contents

- Executive KPI and donut cards for headcount, payroll by role, and attrition.
- Age distribution by department.
- Attrition comparison by tenure, current role, and manager relationship.
- Gender-based attrition age distribution.
- Attrition by job role.
- Bottom KPI row for monthly payroll, payroll growth, and overall rating.
- Sidebar filters for department, job role, gender, marital status, travel, attrition, age, and workplace ratings.
