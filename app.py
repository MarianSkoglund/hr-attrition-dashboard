from __future__ import annotations

from contextlib import contextmanager

import dashboard_core as core


CARD_CAPTIONS = {
    "Total Head Count": "Donut segments show department share.",
    "Age Distribution": "Stacked bars show employee count by age; colors represent departments.",
    "Most Costly Role": "Donut segments show monthly payroll share by job role.",
    "Employee Attrition": "Teal is Attrition No; coral is Attrition Yes.",
    "Attrition Share by Time Dimensions": "Grouped bars compare teal Attrition No with coral Attrition Yes.",
    "Attrition Age Distribution by Gender": "Grouped bars show attrition count by age band; colors represent gender.",
    "Attrition by Job Role": "Bar length shows attrition count; stronger red means higher attrition rate.",
}

_card = core.card
_donut = core.donut
_age_distribution = core.age_distribution
_tenure_bar = core.tenure_bar
_gender_age = core.gender_age
_role_attrition = core.role_attrition


@contextmanager
def card(title: str):
    with _card(title):
        caption = CARD_CAPTIONS.get(title)
        if caption:
            core.st.caption(caption)
        yield


def donut(labels, values, colors, center, label, height=238, legend=False):
    fig = _donut(labels, values, colors, center, label, height=height, legend=True)
    if label == "Employees":
        fig.update_layout(legend_title_text="Department")
    elif label == "Attrition":
        fig.update_layout(legend_title_text="Attrition Status")
    else:
        fig.update_layout(legend_title_text="Job Role")
    return fig


def age_distribution(df):
    fig = _age_distribution(df)
    fig.update_layout(showlegend=True, legend_title_text="Department")
    return fig


def tenure_bar(df, column: str):
    fig = _tenure_bar(df, column)
    fig.update_layout(showlegend=True, legend_title_text="Attrition Status")
    return fig


def gender_age(df):
    fig = _gender_age(df)
    fig.update_layout(showlegend=True, legend_title_text="Gender")
    return fig


def role_attrition(df):
    fig = _role_attrition(df)
    fig.update_layout(
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(
            title="Attrition rate",
            tickformat=".0%",
            thickness=9,
            len=0.72,
            y=0.5,
            titlefont=dict(color=core.MUTED, size=10),
            tickfont=dict(color=core.MUTED, size=9),
        ),
    )
    return fig


core.card = card
core.donut = donut
core.age_distribution = age_distribution
core.tenure_bar = tenure_bar
core.gender_age = gender_age
core.role_attrition = role_attrition


if __name__ == "__main__":
    core.main()
