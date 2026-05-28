from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).parent

TEAL = "#0f9f9a"
DEEP_TEAL = "#0f766e"
CORAL = "#ef6f61"
RED = "#d94f45"
GREEN = "#2f855a"
DARK = "#263238"
MUTED = "#6b7280"
BORDER = "#dfe5e7"
GRID = "#e9eef0"
PALETTE = [TEAL, CORAL, GREEN, "#386fa4", "#f2c14e", "#6d597a", "#8d99ae"]
ATTRITION_COLORS = {"No": TEAL, "Yes": CORAL}

ALIASES = {
    "employee_number": "employee_id",
    "employee_count": "employee_id",
    "employee": "employee_id",
    "id": "employee_id",
    "business_travel": "travel",
    "travel_frequency": "travel",
    "jobrole": "job_role",
    "role": "job_role",
    "monthly_income": "monthly_payroll",
    "monthly_salary": "monthly_payroll",
    "salary": "monthly_payroll",
    "payroll": "monthly_payroll",
    "monthly_payroll_cost": "monthly_payroll_cost",
    "attrition_flag": "attrition",
    "years_with_curr_manager": "years_with_current_manager",
    "years_with_manager": "years_with_current_manager",
    "relationship_satisfaction": "work_relationship_rating",
    "relationship_rating": "work_relationship_rating",
    "work_life_balance": "work_life_balance_rating",
    "worklife_balance": "work_life_balance_rating",
    "environment_satisfaction": "work_environment_rating",
    "environment_rating": "work_environment_rating",
    "job_satisfaction": "overall_rating",
    "overall_satisfaction": "overall_rating",
    "percent_salary_hike": "payroll_growth",
    "salary_hike": "payroll_growth",
}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background: white; color: {DARK}; }}
        .main .block-container {{ max-width: 1560px; padding: 1rem 1.15rem 1.6rem; }}
        [data-testid="stSidebar"] {{ background: #f8fafb; border-right: 1px solid {BORDER}; }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border: 1px solid {BORDER}; border-radius: 10px; background: #fff;
            box-shadow: 0 1px 2px rgba(15,23,42,.04);
        }}
        .page-title {{ font-size: 1.72rem; line-height: 1.15; font-weight: 760; color: {DARK}; margin: 0 0 .12rem; }}
        .page-subtitle {{ color: {MUTED}; font-size: .88rem; margin-bottom: .65rem; }}
        .card-title {{ color: {DARK}; font-size: .93rem; font-weight: 730; margin: -.1rem 0 .18rem; }}
        .metric-tile {{ min-height: 112px; display: flex; flex-direction: column; justify-content: center; gap: .18rem; border-left: 4px solid var(--accent); padding-left: .25rem; }}
        .metric-label {{ color: {MUTED}; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .03em; }}
        .metric-value {{ color: {DARK}; font-size: 2rem; line-height: 1.05; font-weight: 780; }}
        .metric-helper {{ color: {MUTED}; font-size: .8rem; }}
        .metric-delta {{ color: var(--accent); font-size: .82rem; font-weight: 740; }}
        .mini-stat {{ border: 1px solid {BORDER}; border-radius: 8px; padding: .58rem .65rem; background: #fbfcfc; margin-bottom: .48rem; }}
        .mini-stat-label {{ color: {MUTED}; font-size: .76rem; font-weight: 700; }}
        .mini-stat-value {{ color: {DARK}; font-size: 1.35rem; line-height: 1.05; font-weight: 780; }}
        .mini-stat-helper {{ color: {MUTED}; font-size: .76rem; }}
        div[data-testid="stPlotlyChart"] {{ margin-top: -.25rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_name(name: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name).strip())
    text = text.replace("%", " percent ")
    text = re.sub(r"[/\\\-&]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    text = re.sub(r"_+", "_", text)
    return ALIASES.get(text, text)


def normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    names: list[str] = []
    for column in raw.columns:
        target = normalize_name(column)
        seen[target] = seen.get(target, 0) + 1
        names.append(target if seen[target] == 1 else f"{target}_{seen[target]}")
    df = raw.copy()
    df.columns = names
    return df


def clean_label(value: Any) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "Unknown"
    text = str(value).strip()
    lookup = {
        "r&d": "Research & Development",
        "rd": "Research & Development",
        "hr": "Human Resources",
        "travel_rarely": "Travel Rarely",
        "travel frequently": "Travel Frequently",
        "travel_frequently": "Travel Frequently",
        "non_travel": "Non-Travel",
        "non travel": "Non-Travel",
    }
    return lookup.get(text.lower(), text.replace("_", " ").title())


def to_numeric(series: pd.Series, default: float | None = None) -> pd.Series:
    values = pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
    return values.fillna(default) if default is not None else values


def shape_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw)

    if "employee_id" not in df:
        df["employee_id"] = np.arange(1, len(df) + 1)
    if "monthly_payroll_cost" in df:
        df["monthly_payroll"] = df["monthly_payroll_cost"]

    for col in ["gender", "department", "job_role", "marital_status", "travel"]:
        if col not in df:
            df[col] = "Unknown"
        df[col] = df[col].apply(clean_label)

    if "attrition" not in df:
        df["attrition"] = "No"
    flag = df["attrition"].astype(str).str.strip().str.lower()
    yes = {"yes", "y", "true", "1", "left", "leaver", "attrited", "terminated"}
    df["attrition"] = np.where(flag.isin(yes), "Yes", "No")

    defaults = {
        "age": 38,
        "monthly_payroll": np.nan,
        "years_at_company": 0,
        "years_in_current_role": 0,
        "years_with_current_manager": 0,
        "work_relationship_rating": 3,
        "work_life_balance_rating": 3,
        "work_environment_rating": 3,
        "performance_rating": 3,
        "overall_rating": np.nan,
        "payroll_growth": np.nan,
    }
    for col, default in defaults.items():
        if col not in df:
            df[col] = default
        df[col] = to_numeric(df[col], None)

    if df["monthly_payroll"].isna().all():
        df["monthly_payroll"] = 5600
    medians = df.groupby("job_role")["monthly_payroll"].transform("median")
    df["monthly_payroll"] = df["monthly_payroll"].fillna(medians).fillna(df["monthly_payroll"].median()).fillna(5600)
    df["age"] = df["age"].fillna(df["age"].median()).clip(16, 75).round().astype(int)

    for col in [
        "years_at_company",
        "years_in_current_role",
        "years_with_current_manager",
        "work_relationship_rating",
        "work_life_balance_rating",
        "work_environment_rating",
        "performance_rating",
    ]:
        df[col] = df[col].fillna(df[col].median()).clip(lower=0)

    if df["overall_rating"].isna().all():
        rating_cols = ["work_relationship_rating", "work_life_balance_rating", "work_environment_rating", "performance_rating"]
        df["overall_rating"] = df[rating_cols].mean(axis=1)
    df["overall_rating"] = df["overall_rating"].fillna(df["overall_rating"].median()).clip(1, 5)

    if df["payroll_growth"].dropna().median() > 1.5:
        df["payroll_growth"] = df["payroll_growth"] / 100
    if df["payroll_growth"].isna().all():
        df["payroll_growth"] = (0.028 + (df["overall_rating"] - 3) * 0.007 + df["years_at_company"].clip(0, 10) * 0.0012).clip(0.005, 0.16)
    else:
        df["payroll_growth"] = df["payroll_growth"].fillna(df["payroll_growth"].median()).clip(-1, 1)

    return df


def make_demo_data(rows: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    departments = rng.choice(["Sales", "Research & Development", "Operations", "Finance", "Human Resources"], rows, p=[.31, .34, .18, .10, .07])
    roles = {
        "Sales": ["Sales Executive", "Account Manager", "Sales Representative"],
        "Research & Development": ["Research Scientist", "Laboratory Technician", "Research Director"],
        "Operations": ["Operations Manager", "Process Specialist", "Logistics Coordinator"],
        "Finance": ["Financial Analyst", "Finance Manager", "Controller"],
        "Human Resources": ["HR Specialist", "Recruiter", "HR Manager"],
    }
    job_role = [rng.choice(roles[d]) for d in departments]
    age = np.rint(rng.normal(38, 9, rows)).clip(19, 63).astype(int)
    years = np.minimum(np.rint(rng.gamma(2.0, 3.5, rows)), age - 18).clip(0, 35).astype(int)
    work_life = rng.choice([1, 2, 3, 4, 5], rows, p=[.08, .16, .36, .29, .11])
    environment = rng.choice([1, 2, 3, 4, 5], rows, p=[.07, .14, .34, .33, .12])
    relationship = rng.choice([1, 2, 3, 4, 5], rows, p=[.07, .13, .33, .33, .14])
    performance = rng.choice([2, 3, 4, 5], rows, p=[.08, .52, .32, .08])
    risk = .06 + (age < 30) * .07 + (work_life <= 2) * .06 + (environment <= 2) * .06 + (years <= 1) * .06
    attrition = np.where(rng.random(rows) < np.clip(risk, .02, .35), "Yes", "No")
    payroll = np.rint(rng.normal(7600, 2600, rows).clip(2500, 22000)).astype(int)
    return pd.DataFrame({
        "employee_id": range(1, rows + 1),
        "age": age,
        "gender": rng.choice(["Female", "Male"], rows),
        "department": departments,
        "job_role": job_role,
        "monthly_payroll": payroll,
        "attrition": attrition,
        "years_at_company": years,
        "years_in_current_role": np.minimum(years, rng.integers(0, 9, rows)),
        "years_with_current_manager": np.minimum(years, rng.integers(0, 9, rows)),
        "marital_status": rng.choice(["Single", "Married", "Divorced"], rows),
        "travel": rng.choice(["Non-Travel", "Travel Rarely", "Travel Frequently"], rows, p=[.16, .66, .18]),
        "work_relationship_rating": relationship,
        "work_life_balance_rating": work_life,
        "work_environment_rating": environment,
        "performance_rating": performance,
        "overall_rating": np.rint((relationship + work_life + environment + performance) / 4).clip(1, 5),
        "payroll_growth": rng.normal(.045, .015, rows).clip(.005, .14),
    })


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, str, bool]:
    for name in ["hr_attrition_demo_data.csv", "hr_attrition_demo_data.xlsx"]:
        path = APP_DIR / name
        if path.exists():
            raw = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
            return shape_data(raw), name, False
    for pattern in ["*attrition*.csv", "*attrition*.xlsx", "*hr*.csv", "*hr*.xlsx"]:
        matches = sorted(p for p in APP_DIR.glob(pattern) if p.is_file() and not p.name.startswith("~$"))
        if matches:
            path = matches[0]
            raw = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
            return shape_data(raw), path.name, False
    return make_demo_data(), "Generated demo data", True


def fmt_count(value: float) -> str:
    return f"{int(round(value)):,}"


def fmt_money(value: float) -> str:
    return f"{value / 1_000_000:.2f}M"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


@contextmanager
def card(title: str):
    with st.container(border=True):
        st.markdown(f"<div class='card-title'>{title}</div>", unsafe_allow_html=True)
        yield


def metric(label: str, value: str, helper: str, accent: str, delta: str | None = None) -> None:
    delta_html = f"<div class='metric-delta'>{delta}</div>" if delta else ""
    st.markdown(
        f"<div class='metric-tile' style='--accent:{accent};'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>{delta_html}<div class='metric-helper'>{helper}</div></div>",
        unsafe_allow_html=True,
    )


def mini_stat(label: str, value: str, helper: str) -> None:
    st.markdown(
        f"<div class='mini-stat'><div class='mini-stat-label'>{label}</div>"
        f"<div class='mini-stat-value'>{value}</div><div class='mini-stat-helper'>{helper}</div></div>",
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=DARK, family="Arial, sans-serif", size=12),
        showlegend=legend,
        legend=dict(orientation="h", y=1.02, x=0, title_text="", font=dict(size=10, color=MUTED)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED, size=10), title_text="")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED, size=10), title_text="")
    return fig


def empty_fig(text: str, height: int) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, x=.5, y=.5, xref="paper", yref="paper", showarrow=False, font=dict(color=MUTED, size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_fig(fig, height)


def donut(labels: list[str], values: list[float], colors: list[str], center: str, label: str, height: int = 238, legend: bool = False) -> go.Figure:
    if not sum(values):
        return empty_fig("No data available", height)
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=.68,
        sort=False,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="percent",
        textposition="inside",
        textfont=dict(color="white", size=11),
    ))
    fig.update_layout(annotations=[dict(text=f"<b>{center}</b><br><span style='color:{MUTED}'>{label}</span>", x=.5, y=.5, showarrow=False, font=dict(color=DARK, size=18))])
    return style_fig(fig, height, legend)


def top_other(series: pd.Series, n: int) -> pd.Series:
    ordered = series.sort_values(ascending=False)
    if len(ordered) <= n:
        return ordered
    return pd.concat([ordered.head(n), pd.Series({"Other": ordered.iloc[n:].sum()})])


def age_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="age", color="department", nbins=16, barmode="stack", color_discrete_sequence=PALETTE)
    fig.update_traces(marker_line_width=0, hovertemplate="<b>Age %{x}</b><br>Employees: %{y:,}<extra></extra>")
    fig.update_layout(bargap=.08)
    return style_fig(fig, 238)


def tenure_bar(df: pd.DataFrame, column: str) -> go.Figure:
    years = df[column].fillna(0).clip(lower=0).round().astype(int)
    if years.max() <= 12:
        order = [str(i) for i in range(int(years.max()) + 1)]
        bucket = years.astype(str)
    else:
        order = ["0", "1", "2", "3", "4-5", "6-8", "9-12", "13-20", "21+"]
        bucket = pd.cut(years, [-1, 0, 1, 2, 3, 5, 8, 12, 20, 100], labels=order, include_lowest=True).astype(str)
    grouped = df.assign(bucket=bucket).groupby(["bucket", "attrition"], observed=False).size().reset_index(name="employees")
    full = pd.MultiIndex.from_product([order, ["No", "Yes"]], names=["bucket", "attrition"])
    grouped = grouped.set_index(["bucket", "attrition"]).reindex(full, fill_value=0).reset_index()
    fig = px.bar(grouped, x="bucket", y="employees", color="attrition", barmode="group", category_orders={"bucket": order, "attrition": ["No", "Yes"]}, color_discrete_map=ATTRITION_COLORS)
    fig.update_traces(marker_line_width=0, hovertemplate="<b>%{x} years</b><br>Employees: %{y:,}<extra></extra>")
    fig.update_layout(bargap=.28, bargroupgap=.08)
    return style_fig(fig, 216)


def gender_age(df: pd.DataFrame) -> go.Figure:
    attrited = df[df["attrition"] == "Yes"].copy()
    if attrited.empty:
        return empty_fig("No attrition records match filters", 282)
    labels = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-60", "61+"]
    attrited["age_band"] = pd.cut(attrited["age"], [17, 24, 29, 34, 39, 44, 49, 54, 60, 80], labels=labels, include_lowest=True)
    grouped = attrited.groupby(["age_band", "gender"], observed=False).size().reset_index(name="employees")
    fig = px.bar(grouped, x="age_band", y="employees", color="gender", barmode="group", category_orders={"age_band": labels}, color_discrete_map={"Female": CORAL, "Male": DEEP_TEAL, "Unknown": "#8d99ae"})
    fig.update_traces(marker_line_width=0, hovertemplate="<b>%{x}</b><br>Attrition: %{y:,}<extra></extra>")
    return style_fig(fig, 282, True)


def role_attrition(df: pd.DataFrame) -> go.Figure:
    summary = df.assign(left=(df["attrition"] == "Yes").astype(int)).groupby("job_role", as_index=False).agg(attrition_count=("left", "sum"), total=("employee_id", "count"))
    if summary.empty:
        return empty_fig("No role data available", 330)
    summary["attrition_rate"] = np.where(summary["total"] > 0, summary["attrition_count"] / summary["total"], 0)
    summary = summary.sort_values(["attrition_count", "attrition_rate", "total"], ascending=True).tail(12)
    fig = px.bar(summary, x="attrition_count", y="job_role", orientation="h", color="attrition_rate", color_continuous_scale=["#fde4df", CORAL, RED], text="attrition_count", custom_data=["total", "attrition_rate"])
    fig.update_traces(marker_line_width=0, texttemplate="%{text:,}", textposition="outside", cliponaxis=False, hovertemplate="<b>%{y}</b><br>Attrition: %{x:,}<br>Total: %{customdata[0]:,}<br>Rate: %{customdata[1]:.1%}<extra></extra>")
    fig.update_layout(coloraxis_showscale=False, bargap=.24)
    return style_fig(fig, 330)


def sidebar_filters(df: pd.DataFrame) -> dict[str, Any]:
    st.sidebar.title("Filters")
    st.sidebar.caption("Selections apply to every card and chart.")
    filters: dict[str, Any] = {}
    for col, label in [
        ("department", "Department"),
        ("job_role", "Job Role"),
        ("gender", "Gender"),
        ("marital_status", "Marital Status"),
        ("travel", "Travel"),
        ("attrition", "Attrition"),
    ]:
        options = sorted(df[col].dropna().unique().tolist())
        filters[col] = st.sidebar.multiselect(label, options, default=options)
    filters["age_range"] = st.sidebar.slider("Age range", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))
    for col, label in [
        ("work_relationship_rating", "Work Relationship Rating"),
        ("work_life_balance_rating", "Work/Life Balance Rating"),
        ("work_environment_rating", "Work Environment Rating"),
    ]:
        options = sorted(df[col].dropna().round().astype(int).unique().tolist())
        filters[col] = st.sidebar.multiselect(label, options, default=options)
    return filters


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col in ["department", "job_role", "gender", "marital_status", "travel", "attrition"]:
        selected = filters.get(col, [])
        if selected:
            out = out[out[col].isin(selected)]
    low, high = filters["age_range"]
    out = out[out["age"].between(low, high)]
    for col in ["work_relationship_rating", "work_life_balance_rating", "work_environment_rating"]:
        selected = filters.get(col, [])
        if selected:
            out = out[out[col].round().astype(int).isin(selected)]
    return out


def render_header(df: pd.DataFrame, source: str, is_demo: bool) -> None:
    attrition_rate = (df["attrition"] == "Yes").mean() if len(df) else 0
    payroll = df["monthly_payroll"].sum() if len(df) else 0
    st.markdown("<div class='page-title'>HR Attrition Management</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='page-subtitle'>Executive workforce risk view | {fmt_count(len(df))} employees | "
        f"{fmt_pct(attrition_rate)} attrition | {fmt_money(payroll)} monthly payroll | Source: {source}</div>",
        unsafe_allow_html=True,
    )
    if is_demo:
        st.sidebar.info("No HR dataset was found, so the app is using generated demo data.")


def render_dashboard(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns([1.02, 1.36, 1.02, 1.02], gap="small")
    with c1:
        with card("Total Head Count"):
            values = top_other(df["department"].value_counts(), 5)
            st.plotly_chart(donut(values.index.tolist(), values.astype(float).tolist(), (PALETTE * 2)[: len(values)], fmt_count(len(df)), "Employees"), use_container_width=True, config={"displayModeBar": False})
    with c2:
        with card("Age Distribution"):
            st.plotly_chart(age_distribution(df), use_container_width=True, config={"displayModeBar": False})
    with c3:
        with card("Most Costly Role"):
            payroll = top_other(df.groupby("job_role")["monthly_payroll"].sum(), 6)
            top = payroll.sort_values(ascending=False)
            st.plotly_chart(donut(payroll.index.tolist(), payroll.astype(float).tolist(), (PALETTE * 2)[: len(payroll)], fmt_money(top.iloc[0]), top.index[0][:22]), use_container_width=True, config={"displayModeBar": False})
    with c4:
        with card("Employee Attrition"):
            counts = df["attrition"].value_counts().reindex(["No", "Yes"], fill_value=0)
            st.plotly_chart(donut(counts.index.tolist(), counts.astype(float).tolist(), [TEAL, CORAL], fmt_count(counts["Yes"]), "Attrition", legend=True), use_container_width=True, config={"displayModeBar": False})

    with card("Attrition Share by Time Dimensions"):
        a, b, c = st.columns(3, gap="small")
        with a:
            st.caption("Years at Company")
            st.plotly_chart(tenure_bar(df, "years_at_company"), use_container_width=True, config={"displayModeBar": False})
        with b:
            st.caption("Years in Current Role")
            st.plotly_chart(tenure_bar(df, "years_in_current_role"), use_container_width=True, config={"displayModeBar": False})
        with c:
            st.caption("Years with Current Manager")
            st.plotly_chart(tenure_bar(df, "years_with_current_manager"), use_container_width=True, config={"displayModeBar": False})

    left, right = st.columns([1.26, 1], gap="small")
    with left:
        with card("Attrition Age Distribution by Gender"):
            chart_col, stat_col = st.columns([4.4, 1.15], gap="small")
            attrited = df[df["attrition"] == "Yes"]
            totals = df["gender"].value_counts()
            left_gender = attrited["gender"].value_counts()
            with chart_col:
                st.plotly_chart(gender_age(df), use_container_width=True, config={"displayModeBar": False})
            with stat_col:
                for gender in ["Female", "Male"]:
                    mini_stat(gender, fmt_count(left_gender.get(gender, 0)), f"{fmt_count(totals.get(gender, 0))} total employees")
    with right:
        with card("Attrition by Job Role"):
            st.plotly_chart(role_attrition(df), use_container_width=True, config={"displayModeBar": False})

    payroll = df["monthly_payroll"].sum()
    growth = df["payroll_growth"].mean()
    rating = df["overall_rating"].mean()
    attrition_rate = (df["attrition"] == "Yes").mean()
    k1, k2, k3 = st.columns(3, gap="small")
    with k1:
        with card("Monthly Payroll"):
            metric("Monthly Payroll", fmt_money(payroll), f"{fmt_money(payroll * 12)} annualized payroll", DEEP_TEAL)
    with k2:
        with card("Payroll Growth"):
            metric("Payroll Growth", fmt_pct(growth), f"Weighted by {fmt_count(len(df))} selected employees", GREEN if growth >= 0 else RED, f"{fmt_pct(growth - attrition_rate * .05)} net workforce signal")
    with k3:
        with card("Overall Rating"):
            metric("Overall Rating", f"{rating:.1f}/5", "Average employee experience score", CORAL if rating < 3 else TEAL)


def main() -> None:
    st.set_page_config(page_title="HR Attrition Management", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    data, source, is_demo = load_data()
    filters = sidebar_filters(data)
    filtered = apply_filters(data, filters)
    render_header(filtered, source, is_demo)
    if filtered.empty:
        st.warning("No employees match the current filters. Relax one or more filters to bring the dashboard back.")
        return
    render_dashboard(filtered)


if __name__ == "__main__":
    main()
