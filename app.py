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
TEAL = '#0f9f9a'
DEEP_TEAL = '#0f766e'
CORAL = '#ef6f61'
RED = '#d94f45'
GREEN = '#2f855a'
DARK = '#263238'
MUTED = '#6b7280'
BORDER = '#dfe5e7'
GRID = '#e9eef0'
ATTRITION_COLORS = {'No': TEAL, 'Yes': CORAL}
PALETTE = [TEAL, CORAL, GREEN, '#386fa4', '#f2c14e', '#6d597a', '#8d99ae']
ALIASES = {
    'employee_number': 'employee_id',
    'employee_count': 'employee_id',
    'employee': 'employee_id',
    'id': 'employee_id',
    'business_travel': 'travel',
    'travel_frequency': 'travel',
    'jobrole': 'job_role',
    'role': 'job_role',
    'monthly_income': 'monthly_payroll',
    'monthly_salary': 'monthly_payroll',
    'monthly_rate': 'monthly_payroll',
    'salary': 'monthly_payroll',
    'payroll': 'monthly_payroll',
    'relationship_satisfaction': 'work_relationship_rating',
    'relationship_rating': 'work_relationship_rating',
    'work_relationship': 'work_relationship_rating',
    'work_life_balance': 'work_life_balance_rating',
    'worklife_balance': 'work_life_balance_rating',
    'environment_satisfaction': 'work_environment_rating',
    'environment_rating': 'work_environment_rating',
    'work_environment': 'work_environment_rating',
    'job_satisfaction': 'overall_rating',
    'overall_satisfaction': 'overall_rating',
    'rating': 'overall_rating',
    'percent_salary_hike': 'payroll_growth',
    'salary_hike': 'payroll_growth',
}


def css() -> None:
    st.markdown(f'''
    <style>
    .stApp {{ background: white; color: {DARK}; }}
    .main .block-container {{ max-width: 1560px; padding: 1rem 1.15rem 1.6rem; }}
    [data-testid='stSidebar'] {{ background: #f8fafb; border-right: 1px solid {BORDER}; }}
    [data-testid='stVerticalBlockBorderWrapper'] {{ border: 1px solid {BORDER}; border-radius: 10px; background: white; box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
    .page-title {{ font-size: 1.72rem; line-height: 1.15; font-weight: 760; color: {DARK}; margin: 0 0 .12rem; }}
    .page-subtitle {{ color: {MUTED}; font-size: .88rem; margin-bottom: .65rem; }}
    .card-title {{ color: {DARK}; font-size: .93rem; font-weight: 730; margin: -.1rem 0 .18rem; }}
    .metric-tile {{ min-height: 112px; display: flex; flex-direction: column; justify-content: center; gap: .18rem; border-left: 4px solid var(--accent); padding-left: .2rem; }}
    .metric-label {{ color: {MUTED}; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .03em; }}
    .metric-value {{ color: {DARK}; font-size: 2rem; line-height: 1.05; font-weight: 780; }}
    .metric-helper {{ color: {MUTED}; font-size: .8rem; }}
    .metric-delta {{ color: var(--accent); font-size: .82rem; font-weight: 740; }}
    .mini-stat {{ border: 1px solid {BORDER}; border-radius: 8px; padding: .58rem .65rem; background: #fbfcfc; margin-bottom: .48rem; }}
    .mini-stat-label {{ color: {MUTED}; font-size: .76rem; font-weight: 700; }}
    .mini-stat-value {{ color: {DARK}; font-size: 1.35rem; line-height: 1.05; font-weight: 780; }}
    .mini-stat-helper {{ color: {MUTED}; font-size: .76rem; }}
    div[data-testid='stPlotlyChart'] {{ margin-top: -.25rem; }}
    </style>
    ''', unsafe_allow_html=True)


def norm(name: Any) -> str:
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', str(name).strip())
    text = text.replace('%', ' percent ')
    text = re.sub(r'[/\\\-&]+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_').lower()
    return ALIASES.get(re.sub(r'_+', '_', text), re.sub(r'_+', '_', text))


def clean_label(value: Any) -> str:
    if pd.isna(value) or str(value).strip() == '':
        return 'Unknown'
    text = str(value).strip().replace('_', ' ')
    lookup = {
        'r&d': 'Research & Development',
        'rd': 'Research & Development',
        'hr': 'Human Resources',
        'travel rarely': 'Travel Rarely',
        'travel frequently': 'Travel Frequently',
        'non travel': 'Non-Travel',
    }
    return lookup.get(text.lower(), text.title())


def numeric(series: pd.Series, default: float | None = None) -> pd.Series:
    values = pd.to_numeric(series.astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')
    return values.fillna(default) if default is not None else values


def demo_data(rows: int = 1260) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    departments = np.array(['Sales', 'Research & Development', 'Operations', 'Finance', 'Human Resources'])
    department = rng.choice(departments, size=rows, p=[.31, .34, .18, .10, .07])
    role_map = {
        'Sales': ['Sales Executive', 'Account Manager', 'Sales Representative', 'Sales Director'],
        'Research & Development': ['Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Research Director'],
        'Operations': ['Operations Manager', 'Process Specialist', 'Plant Supervisor', 'Logistics Coordinator'],
        'Finance': ['Financial Analyst', 'Finance Manager', 'Payroll Specialist', 'Controller'],
        'Human Resources': ['HR Specialist', 'Recruiter', 'HR Manager', 'People Partner'],
    }
    role_base = {
        'Sales Executive': 7800, 'Account Manager': 8400, 'Sales Representative': 4200, 'Sales Director': 14500,
        'Research Scientist': 6500, 'Laboratory Technician': 4100, 'Manufacturing Director': 11900, 'Research Director': 15100,
        'Operations Manager': 10200, 'Process Specialist': 5900, 'Plant Supervisor': 6900, 'Logistics Coordinator': 4600,
        'Financial Analyst': 7200, 'Finance Manager': 11800, 'Payroll Specialist': 5100, 'Controller': 13500,
        'HR Specialist': 5300, 'Recruiter': 5700, 'HR Manager': 9800, 'People Partner': 7600,
    }
    role = np.array([rng.choice(role_map[d]) for d in department])
    gender = rng.choice(['Female', 'Male'], size=rows, p=[.48, .52])
    marital = rng.choice(['Single', 'Married', 'Divorced'], size=rows, p=[.34, .49, .17])
    travel = rng.choice(['Non-Travel', 'Travel Rarely', 'Travel Frequently'], size=rows, p=[.16, .66, .18])
    age = np.rint(rng.normal(38.5, 9.8, rows)).clip(19, 63).astype(int)
    years = np.minimum(np.rint(rng.gamma(2.1, 3.7, rows)), age - 18).clip(0, 36).astype(int)
    role_years = np.minimum(years, np.rint(rng.gamma(1.65, 2.4, rows))).astype(int)
    manager_years = np.minimum(years, np.rint(rng.gamma(1.55, 2.5, rows))).astype(int)
    relationship = rng.choice([1, 2, 3, 4, 5], rows, p=[.07, .13, .33, .33, .14])
    work_life = rng.choice([1, 2, 3, 4, 5], rows, p=[.08, .16, .36, .29, .11])
    environment = rng.choice([1, 2, 3, 4, 5], rows, p=[.07, .14, .34, .33, .12])
    performance = rng.choice([1, 2, 3, 4, 5], rows, p=[.02, .08, .48, .34, .08])
    rating = np.rint((relationship + work_life + environment + performance) / 4 + rng.normal(0, .22, rows)).clip(1, 5)
    base_pay = np.array([role_base[r] for r in role])
    payroll = (base_pay * (1 + (age - 38) * .012) * (1 + years.clip(0, 18) * .018) * (1 + (performance - 3) * .045) + rng.normal(0, 650, rows)).clip(2800, 24000)
    risk = .055 + (age < 30) * .075 + (travel == 'Travel Frequently') * .06 + (marital == 'Single') * .035 + (work_life <= 2) * .065 + (environment <= 2) * .055 + (years <= 1) * .07 - (rating >= 4) * .035 - (years >= 8) * .025
    attrition = np.where(rng.random(rows) < np.clip(risk, .02, .39), 'Yes', 'No')
    growth = (.027 + (performance - 3) * .007 + rng.normal(.014, .012, rows)).clip(.005, .16)
    return pd.DataFrame({
        'employee_id': np.arange(10001, 10001 + rows), 'age': age, 'gender': gender, 'department': department,
        'job_role': role, 'monthly_payroll': np.rint(payroll).astype(int), 'attrition': attrition,
        'years_at_company': years, 'years_in_current_role': role_years, 'years_with_current_manager': manager_years,
        'marital_status': marital, 'travel': travel, 'work_relationship_rating': relationship,
        'work_life_balance_rating': work_life, 'work_environment_rating': environment, 'performance_rating': performance,
        'overall_rating': rating.astype(int), 'payroll_growth': growth,
    })


def shape_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={c: norm(c) for c in raw.columns}).copy()
    if 'employee_id' not in df:
        df['employee_id'] = np.arange(1, len(df) + 1)
    for col in ['gender', 'department', 'job_role', 'marital_status', 'travel']:
        if col not in df:
            df[col] = 'Unknown'
        df[col] = df[col].apply(clean_label)
    if 'attrition' not in df:
        df['attrition'] = 'No'
    flag = df['attrition'].astype(str).str.strip().str.lower()
    df['attrition'] = np.where(flag.isin(['yes', 'y', 'true', '1', 'left', 'leaver', 'attrited', 'terminated']), 'Yes', 'No')
    defaults = {
        'age': 38, 'monthly_payroll': np.nan, 'years_at_company': 0, 'years_in_current_role': 0,
        'years_with_current_manager': 0, 'work_relationship_rating': 3, 'work_life_balance_rating': 3,
        'work_environment_rating': 3, 'performance_rating': 3, 'overall_rating': np.nan, 'payroll_growth': np.nan,
    }
    for col, default in defaults.items():
        if col not in df:
            df[col] = default
        df[col] = numeric(df[col], None)
    if df['monthly_payroll'].isna().all():
        df['monthly_payroll'] = 5600
    df['monthly_payroll'] = df['monthly_payroll'].fillna(df.groupby('job_role')['monthly_payroll'].transform('median')).fillna(df['monthly_payroll'].median())
    df['age'] = df['age'].fillna(df['age'].median()).clip(16, 75).round().astype(int)
    for col in ['years_at_company', 'years_in_current_role', 'years_with_current_manager', 'work_relationship_rating', 'work_life_balance_rating', 'work_environment_rating', 'performance_rating']:
        df[col] = df[col].fillna(df[col].median()).clip(lower=0)
    if df['overall_rating'].isna().all():
        df['overall_rating'] = df[['work_relationship_rating', 'work_life_balance_rating', 'work_environment_rating', 'performance_rating']].mean(axis=1)
    df['overall_rating'] = df['overall_rating'].fillna(df['overall_rating'].median()).clip(1, 5)
    if df['payroll_growth'].dropna().median() > 1.5:
        df['payroll_growth'] = df['payroll_growth'] / 100
    if df['payroll_growth'].isna().all():
        df['payroll_growth'] = (.028 + (df['overall_rating'] - 3) * .007 + df['years_at_company'].clip(0, 10) * .0012).clip(.005, .16)
    return df


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, str, bool]:
    for name in ['hr_attrition_demo_data.csv', 'hr_attrition_demo_data.xlsx']:
        path = APP_DIR / name
        if path.exists():
            raw = pd.read_csv(path) if path.suffix == '.csv' else pd.read_excel(path)
            return shape_data(raw), name, False
    for pattern in ['*attrition*.csv', '*attrition*.xlsx', '*hr*.csv', '*hr*.xlsx']:
        found = sorted(APP_DIR.glob(pattern))
        if found:
            path = found[0]
            raw = pd.read_csv(path) if path.suffix == '.csv' else pd.read_excel(path)
            return shape_data(raw), path.name, False
    return demo_data(), 'Generated demo data', True


def count_fmt(value: float) -> str:
    return f'{int(round(value)):,}'


def money_fmt(value: float) -> str:
    return f'{value / 1_000_000:.2f}M'


def pct_fmt(value: float) -> str:
    return f'{value * 100:.1f}%'


@contextmanager
def card(title: str):
    with st.container(border=True):
        st.markdown(f'<div class=card-title>{title}</div>', unsafe_allow_html=True)
        yield


def metric(label: str, value: str, helper: str, accent: str = TEAL, delta: str | None = None) -> None:
    delta_html = f'<div class=metric-delta>{delta}</div>' if delta else ''
    st.markdown(f'<div class=metric-tile style=--accent:{accent};><div class=metric-label>{label}</div><div class=metric-value>{value}</div>{delta_html}<div class=metric-helper>{helper}</div></div>', unsafe_allow_html=True)


def mini(label: str, value: str, helper: str) -> None:
    st.markdown(f'<div class=mini-stat><div class=mini-stat-label>{label}</div><div class=mini-stat-value>{value}</div><div class=mini-stat-helper>{helper}</div></div>', unsafe_allow_html=True)


def style(fig: go.Figure, height: int, legend: bool = False) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=DARK, family='Arial, sans-serif', size=12), showlegend=legend, legend=dict(orientation='h', y=1.02, x=0, font=dict(size=10, color=MUTED), title_text=''))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED, size=10), title_text='')
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED, size=10), title_text='')
    return fig


def empty_fig(text: str, height: int) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, x=.5, y=.5, xref='paper', yref='paper', showarrow=False, font=dict(color=MUTED, size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height)


def donut(labels: list[str], values: list[float], colors: list[str], center: str, label: str, height: int = 238, legend: bool = False) -> go.Figure:
    if not sum(values):
        return empty_fig('No data available', height)
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=.68, sort=False, marker=dict(colors=colors, line=dict(color='white', width=2)), textinfo='percent', textposition='inside', textfont=dict(color='white', size=11)))
    fig.update_layout(annotations=[dict(text=f'<b>{center}</b><br><span style=color:{MUTED}>{label}</span>', x=.5, y=.5, showarrow=False, font=dict(color=DARK, size=18))])
    return style(fig, height, legend)


def top_other(series: pd.Series, n: int) -> pd.Series:
    values = series.sort_values(ascending=False)
    if len(values) <= n:
        return values
    return pd.concat([values.head(n), pd.Series({'Other': values.iloc[n:].sum()})])


def age_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x='age', color='department', nbins=16, barmode='stack', color_discrete_sequence=PALETTE)
    fig.update_traces(marker_line_width=0, hovertemplate='<b>Age %{x}</b><br>Employees: %{y:,}<extra></extra>')
    fig.update_layout(bargap=.08)
    return style(fig, 238)


def tenure_chart(df: pd.DataFrame, col: str) -> go.Figure:
    years = df[col].fillna(0).clip(lower=0).round().astype(int)
    if years.max() <= 12:
        bucket = years.astype(str)
        order = [str(i) for i in range(int(years.max()) + 1)]
    else:
        order = ['0', '1', '2', '3', '4-5', '6-8', '9-12', '13-20', '21+']
        bucket = pd.cut(years, [-1, 0, 1, 2, 3, 5, 8, 12, 20, 100], labels=order, include_lowest=True).astype(str)
    grouped = df.assign(bucket=bucket).groupby(['bucket', 'attrition'], observed=False).size().reset_index(name='employees')
    full = pd.MultiIndex.from_product([order, ['No', 'Yes']], names=['bucket', 'attrition'])
    grouped = grouped.set_index(['bucket', 'attrition']).reindex(full, fill_value=0).reset_index()
    fig = px.bar(grouped, x='bucket', y='employees', color='attrition', barmode='group', category_orders={'bucket': order, 'attrition': ['No', 'Yes']}, color_discrete_map=ATTRITION_COLORS)
    fig.update_traces(marker_line_width=0, hovertemplate='<b>%{x} years</b><br>Employees: %{y:,}<extra></extra>')
    fig.update_layout(bargap=.28, bargroupgap=.08)
    return style(fig, 216)


def gender_age_chart(df: pd.DataFrame) -> go.Figure:
    left = df[df['attrition'] == 'Yes'].copy()
    if left.empty:
        return empty_fig('No attrition records match filters', 282)
    labels = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-60', '61+']
    left['age_band'] = pd.cut(left['age'], [17, 24, 29, 34, 39, 44, 49, 54, 60, 80], labels=labels, include_lowest=True)
    grouped = left.groupby(['age_band', 'gender'], observed=False).size().reset_index(name='employees')
    fig = px.bar(grouped, x='age_band', y='employees', color='gender', barmode='group', category_orders={'age_band': labels}, color_discrete_map={'Female': CORAL, 'Male': DEEP_TEAL})
    fig.update_traces(marker_line_width=0, hovertemplate='<b>%{x}</b><br>Attrition: %{y:,}<extra></extra>')
    return style(fig, 282, True)


def role_chart(df: pd.DataFrame) -> go.Figure:
    summary = df.assign(left=(df['attrition'] == 'Yes').astype(int)).groupby('job_role', as_index=False).agg(attrition_count=('left', 'sum'), total=('employee_id', 'count'))
    summary['attrition_rate'] = np.where(summary['total'] > 0, summary['attrition_count'] / summary['total'], 0)
    summary = summary.sort_values(['attrition_count', 'attrition_rate', 'total']).tail(12)
    fig = px.bar(summary, x='attrition_count', y='job_role', orientation='h', color='attrition_rate', color_continuous_scale=['#fde4df', CORAL, RED], text='attrition_count', custom_data=['total', 'attrition_rate'])
    fig.update_traces(marker_line_width=0, texttemplate='%{text:,}', textposition='outside', cliponaxis=False, hovertemplate='<b>%{y}</b><br>Attrition: %{x:,}<br>Total: %{customdata[0]:,}<br>Rate: %{customdata[1]:.1%}<extra></extra>')
    fig.update_layout(coloraxis_showscale=False, bargap=.24)
    return style(fig, 330)


def sidebar(df: pd.DataFrame) -> dict[str, Any]:
    st.sidebar.title('Filters')
    st.sidebar.caption('Selections apply to every card and chart.')
    filters: dict[str, Any] = {}
    for col, label in [('department', 'Department'), ('job_role', 'Job Role'), ('gender', 'Gender'), ('marital_status', 'Marital Status'), ('travel', 'Travel'), ('attrition', 'Attrition')]:
        options = sorted(df[col].dropna().unique().tolist())
        filters[col] = st.sidebar.multiselect(label, options, default=options)
    filters['age_range'] = st.sidebar.slider('Age range', int(df['age'].min()), int(df['age'].max()), (int(df['age'].min()), int(df['age'].max())))
    for col, label in [('work_relationship_rating', 'Work Relationship Rating'), ('work_life_balance_rating', 'Work/Life Balance Rating'), ('work_environment_rating', 'Work Environment Rating')]:
        options = sorted(df[col].dropna().round().astype(int).unique().tolist())
        filters[col] = st.sidebar.multiselect(label, options, default=options)
    return filters


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col in ['department', 'job_role', 'gender', 'marital_status', 'travel', 'attrition']:
        if filters[col]:
            out = out[out[col].isin(filters[col])]
    low, high = filters['age_range']
    out = out[out['age'].between(low, high)]
    for col in ['work_relationship_rating', 'work_life_balance_rating', 'work_environment_rating']:
        if filters[col]:
            out = out[out[col].round().astype(int).isin(filters[col])]
    return out


def header(df: pd.DataFrame, source: str, demo: bool) -> None:
    rate = (df['attrition'] == 'Yes').mean() if len(df) else 0
    payroll = df['monthly_payroll'].sum() if len(df) else 0
    st.markdown('<div class=page-title>HR Attrition Management</div>', unsafe_allow_html=True)
    st.markdown(f'<div class=page-subtitle>Executive workforce risk view | {count_fmt(len(df))} employees | {pct_fmt(rate)} attrition | {money_fmt(payroll)} monthly payroll | Source: {source}</div>', unsafe_allow_html=True)
    if demo:
        st.sidebar.info('No HR dataset file was found, so this public demo uses generated data. Add hr_attrition_demo_data.csv or .xlsx to the repo to use the provided file.')


def dashboard(df: pd.DataFrame) -> None:
    a, b, c, d = st.columns([1.02, 1.36, 1.02, 1.02], gap='small')
    with a, card('Total Head Count'):
        values = top_other(df['department'].value_counts(), 5)
        st.plotly_chart(donut(values.index.tolist(), values.astype(float).tolist(), (PALETTE * 2)[:len(values)], count_fmt(len(df)), 'Employees'), use_container_width=True, config={'displayModeBar': False})
    with b, card('Age Distribution'):
        st.plotly_chart(age_chart(df), use_container_width=True, config={'displayModeBar': False})
    with c, card('Most Costly Role'):
        payroll = top_other(df.groupby('job_role')['monthly_payroll'].sum(), 6)
        top = payroll.sort_values(ascending=False)
        st.plotly_chart(donut(payroll.index.tolist(), payroll.astype(float).tolist(), (PALETTE * 2)[:len(payroll)], money_fmt(top.iloc[0]), top.index[0][:22]), use_container_width=True, config={'displayModeBar': False})
    with d, card('Employee Attrition'):
        counts = df['attrition'].value_counts().reindex(['No', 'Yes'], fill_value=0)
        st.plotly_chart(donut(counts.index.tolist(), counts.astype(float).tolist(), [TEAL, CORAL], count_fmt(counts['Yes']), 'Attrition', legend=True), use_container_width=True, config={'displayModeBar': False})

    with card('Attrition Share by Time Dimensions'):
        x, y, z = st.columns(3, gap='small')
        with x:
            st.caption('Years at Company')
            st.plotly_chart(tenure_chart(df, 'years_at_company'), use_container_width=True, config={'displayModeBar': False})
        with y:
            st.caption('Years in Current Role')
            st.plotly_chart(tenure_chart(df, 'years_in_current_role'), use_container_width=True, config={'displayModeBar': False})
        with z:
            st.caption('Years with Current Manager')
            st.plotly_chart(tenure_chart(df, 'years_with_current_manager'), use_container_width=True, config={'displayModeBar': False})

    left, right = st.columns([1.26, 1], gap='small')
    with left, card('Attrition Age Distribution by Gender'):
        chart_col, stat_col = st.columns([4.4, 1.15], gap='small')
        attrited = df[df['attrition'] == 'Yes']
        with chart_col:
            st.plotly_chart(gender_age_chart(df), use_container_width=True, config={'displayModeBar': False})
        with stat_col:
            totals = df['gender'].value_counts()
            left_gender = attrited['gender'].value_counts()
            for gender in ['Female', 'Male']:
                mini(gender, count_fmt(left_gender.get(gender, 0)), f'{count_fmt(totals.get(gender, 0))} total employees')
    with right, card('Attrition by Job Role'):
        st.plotly_chart(role_chart(df), use_container_width=True, config={'displayModeBar': False})

    payroll = df['monthly_payroll'].sum()
    growth = df['payroll_growth'].mean()
    rating = df['overall_rating'].mean()
    rate = (df['attrition'] == 'Yes').mean()
    m1, m2, m3 = st.columns(3, gap='small')
    with m1, card('Monthly Payroll'):
        metric('Monthly Payroll', money_fmt(payroll), f'{money_fmt(payroll * 12)} annualized payroll', DEEP_TEAL)
    with m2, card('Payroll Growth'):
        metric('Payroll Growth', pct_fmt(growth), f'Weighted by {count_fmt(len(df))} selected employees', GREEN if growth >= 0 else RED, f'{pct_fmt(growth - rate * .05)} net workforce signal')
    with m3, card('Overall Rating'):
        metric('Overall Rating', f'{rating:.1f}/5', 'Average employee experience score', CORAL if rating < 3 else TEAL)


def main() -> None:
    st.set_page_config(page_title='HR Attrition Management', layout='wide', initial_sidebar_state='expanded')
    css()
    data, source, is_demo = load_data()
    filters = sidebar(data)
    filtered = apply_filters(data, filters)
    header(filtered, source, is_demo)
    if filtered.empty:
        st.warning('No employees match the current filters. Relax one or more filters to bring the dashboard back.')
        return
    dashboard(filtered)


if __name__ == '__main__':
    main()
