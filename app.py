import os
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

# --------------------------------------------------------
# BASIC CONFIG
# --------------------------------------------------------
st.set_page_config(
    page_title="Protection Indicators – Evaluation Dashboard",
    layout="wide",
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["1. Dashboard", "2. Interview form"],
)

st.sidebar.markdown("---")
st.sidebar.write(
    "This tool uses the evaluation data you already collected with Kobo. "
    "It is for internal analysis, reflection and learning."
)

# --------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the Excel file exported from Kobo.
    The file must be in the same folder as app.py.
    """
    candidates = [
        "evaluation_protection_indicators_PRM_VF_-_all_versions_-_labels_-_2024-10-15-11-16-37.xlsx",
        "protection_evaluation.xlsx",
    ]

    path = None
    for cand in candidates:
        if os.path.exists(cand):
            path = cand
            break

    if path is None:
        st.error(
            "No data file found. Please upload either "
            "'evaluation_protection_indicators_PRM_VF_-_all_versions_-_labels_-_2024-10-15-11-16-37.xlsx' "
            "or 'protection_evaluation.xlsx' to the same folder as app.py."
        )
        st.stop()

    df = pd.read_excel(path)

    # Try to standardise age to numeric
    age_col = next((c for c in df.columns if "Age" in c or "Âge" in c or "age" in c), None)
    if age_col is not None:
        df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

    return df


data = load_data()

# Detect key columns dynamically to be robust to small naming changes
name_col = next((c for c in data.columns if c.strip().startswith("Nom")), None)
camp_col = next((c for c in data.columns if "Camps" in c), None)
sex_col = next((c for c in data.columns if "Sexe" in c or "Sex" in c), None)
age_col = next((c for c in data.columns if "Age" in c or "Âge" in c or "age" in c), None)
eth_col = next((c for c in data.columns if "Ethnie" in c or "Ethnic" in c), None)
pbs_col = next((c for c in data.columns if "PBS" in c), None)

# Identify indicator columns: survey questions with limited distinct answers
base_cols = {name_col, camp_col, sex_col, age_col, eth_col, pbs_col}
base_cols = {c for c in base_cols if c is not None}
candidate_cols = [c for c in data.columns if c not in base_cols and not c.startswith("_")]

indicator_cols = []
for c in candidate_cols:
    non_null = data[c].dropna()
    if non_null.empty:
        continue
    nunique = non_null.nunique()
    # Likert type or categorical questions, not free text comments
    if 1 < nunique <= 15:
        indicator_cols.append(c)

# Helper for shortening long labels
def short(text: str, n: int = 90) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


# --------------------------------------------------------
# PAGE 1 – DASHBOARD
# --------------------------------------------------------
if page.startswith("1"):
    st.title("Protection indicators – evaluation dashboard")

    # ---------------------- FILTERS ----------------------
    st.subheader("Filters")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    df = data.copy()

    # Camp filter
    with col_f1:
        if camp_col is not None:
            camps = sorted(df[camp_col].dropna().unique())
            camp_filter = st.multiselect("Camp", camps, default=camps)
            df = df[df[camp_col].isin(camp_filter)]
        else:
            st.info("Camp column not found in dataset.")

    # Sex filter
    with col_f2:
        if sex_col is not None:
            sexes = sorted(df[sex_col].dropna().unique())
            sex_filter = st.multiselect("Sex", sexes, default=sexes)
            df = df[df[sex_col].isin(sex_filter)]
        else:
            st.info("Sex column not found in dataset.")

    # Age filter
    with col_f3:
        if age_col is not None and df[age_col].notna().any():
            min_age = int(df[age_col].min())
            max_age = int(df[age_col].max())
            age_range = st.slider(
                "Age range",
                min_value=min_age,
                max_value=max_age,
                value=(min_age, max_age),
            )
            df = df[df[age_col].between(age_range[0], age_range[1])]
        else:
            st.info("Age column not available or empty.")

    # Disability filter
    with col_f4:
        if pbs_col is not None:
            pbs_vals = sorted(df[pbs_col].dropna().astype(str).unique())
            pbs_filter = st.multiselect("PBS (disability)", pbs_vals, default=pbs_vals)
            df = df[df[pbs_col].astype(str).isin(pbs_filter)]
        else:
            st.info("PBS column not found in dataset.")

    st.caption(f"Number of interviews after filters: {len(df)}")

    if len(df) == 0:
        st.warning("No records match these filters. Please adjust the selection.")
        st.stop()

    # ---------------------- KEY FIGURES ----------------------
    st.subheader("Key figures")

    col_k1, col_k2, col_k3, col_k4 = st.columns(4)

    with col_k1:
        st.metric("Total respondents", len(df))

    with col_k2:
        if age_col is not None and df[age_col].notna().any():
            st.metric("Average age", round(df[age_col].mean(), 1))
        else:
            st.metric("Average age", "N/A")

    with col_k3:
        if sex_col is not None:
            sex_counts = df[sex_col].value_counts()
            m = sex_counts.get("Masculin", 0)
            f = sex_counts.get("Féminin", 0)
            st.metric("Sex ratio (M:F)", f"{m}:{f}")
        else:
            st.metric("Sex ratio", "N/A")

    with col_k4:
        if pbs_col is not None:
            pbs_yes = df[pbs_col].astype(str).str.contains(
                "Oui", case=False, na=False
            ).mean()
            st.metric(
                "Persons with disability (self reported, percent)",
                f"{round(pbs_yes * 100, 1)} %",
            )
        else:
            st.metric("Persons with disability", "N/A")

    # ---------------------- DEMOGRAPHICS ----------------------
    st.subheader("Demographic profile")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        if sex_col is not None:
            st.write("Sex distribution")
            st.bar_chart(df[sex_col].value_counts())
        else:
            st.info("Sex column not available in this dataset.")

    with col_d2:
        if age_col is not None and df[age_col].notna().any():
            st.write("Age distribution")
            age_chart = (
                alt.Chart(df.dropna(subset=[age_col]))
                .mark_bar()
                .encode(
                    alt.X(f"{age_col}:Q", bin=alt.Bin(maxbins=15), title="Age"),
                    alt.Y("count()", title="Number of persons"),
                )
                .properties(height=300)
            )
            st.altair_chart(age_chart, use_container_width=True)
        else:
            st.info("Age column not available or empty.")

    col_d3, col_d4 = st.columns(2)

    with col_d3:
        if camp_col is not None:
            st.write("Distribution by camp")
            st.bar_chart(df[camp_col].value_counts())
        else:
            st.info("Camp column not available.")

    with col_d4:
        if eth_col is not None:
            st.write("Ethnicity distribution")
            st.bar_chart(df[eth_col].value_counts())
        else:
            st.info("Ethnicity column not available.")

    # ---------------------- INDICATORS ----------------------
    st.subheader("Protection indicators")

    if not indicator_cols:
        st.warning(
            "Could not automatically detect indicator columns. "
            "Please review the column selection logic in the code."
        )
    else:
        # Choose an indicator to explore in depth
        selected_indicator = st.selectbox(
            "Select an indicator",
            options=indicator_cols,
            format_func=lambda c: short(c),
        )

        st.markdown(f"**Question:** {selected_indicator}")

        # Optional filter by answer for this indicator
        answer_vals = sorted(df[selected_indicator].dropna().astype(str).unique())
        answer_filter = st.multiselect(
            "Filter by answer (optional)",
            options=answer_vals,
            default=answer_vals,
        )
        df_ind = df[df[selected_indicator].astype(str).isin(answer_filter)]

        st.caption(f"Number of interviews for this indicator after answer filter: {len(df_ind)}")

        # Response distribution
        counts = df_ind[selected_indicator].astype(str).value_counts()
        resp_chart = (
            alt.Chart(counts.reset_index())
            .mark_bar()
            .encode(
                x=alt.X("index:N", title="Response"),
                y=alt.Y(selected_indicator + ":Q", field="count", title="Number of respondents"),
            )
            .properties(height=300)
        )
        # Work around altair naming
        resp_chart = alt.Chart(counts.reset_index().rename(columns={"index": "Response", selected_indicator: "Count"})).mark_bar().encode(
            x=alt.X("Response:N", title="Response"),
            y=alt.Y("Count:Q", title="Number of respondents"),
        )
        st.altair_chart(resp_chart, use_container_width=True)

        # Map common Likert answers to numeric to get simple scores
        likert_map = {
            "Oui, complètement": 4,
            "Plutôt oui": 3,
            "Pas vraiment": 2,
            "Pas du tout": 1,
        }
        scores = df_ind[selected_indicator].map(likert_map)
        if scores.notna().any():
            df_ind = df_ind.copy()
            df_ind["__score__"] = scores

            st.subheader("Summary by camp (higher score usually more positive)")

            if camp_col is not None:
                by_camp = df_ind.groupby(camp_col)["__score__"].mean().dropna().sort_values(ascending=False)
                if not by_camp.empty:
                    st.bar_chart(by_camp)
                else:
                    st.info("No numeric scores could be calculated for this indicator.")
            else:
                st.info("Camp column not available for summary by camp.")

        # Cross analysis by sex
        st.subheader("Cross analysis: sex x indicator")

        if sex_col is not None:
            cross_sex = (
                df_ind.groupby([sex_col, selected_indicator])
                .size()
                .reset_index(name="count")
            )
            if not cross_sex.empty:
                cross_chart_sex = (
                    alt.Chart(cross_sex)
                    .mark_bar()
                    .encode(
                        x=alt.X(f"{selected_indicator}:N", title="Response"),
                        y=alt.Y("count:Q", title="Number"),
                        color=sex_col + ":N",
                        column=sex_col + ":N",
                    )
                )
                st.altair_chart(cross_chart_sex, use_container_width=True)
            else:
                st.info("No data for this cross analysis.")
        else:
            st.info("Sex column not available for cross analysis.")

        # Cross analysis by camp
        st.subheader("Cross analysis: camp x indicator")

        if camp_col is not None:
            cross_camp = (
                df_ind.groupby([camp_col, selected_indicator])
                .size()
                .reset_index(name="count")
            )
            if not cross_camp.empty:
                cross_chart_camp = (
                    alt.Chart(cross_camp)
                    .mark_bar()
                    .encode(
                        x=alt.X(f"{selected_indicator}:N", title="Response"),
                        y=alt.Y("count:Q", title="Number"),
                        color=camp_col + ":N",
                        column=camp_col + ":N",
                    )
                )
                st.altair_chart(cross_chart_camp, use_container_width=True)
            else:
                st.info("No data for this cross analysis.")
        else:
            st.info("Camp column not available for cross analysis.")

        # Cross analysis by disability
        st.subheader("Cross analysis: disability (PBS) x indicator")

        if pbs_col is not None:
            cross_pbs = (
                df_ind.groupby([pbs_col, selected_indicator])
                .size()
                .reset_index(name="count")
            )
            if not cross_pbs.empty:
                cross_chart_pbs = (
                    alt.Chart(cross_pbs)
                    .mark_bar()
                    .encode(
                        x=alt.X(f"{selected_indicator}:N", title="Response"),
                        y=alt.Y("count:Q", title="Number"),
                        color=pbs_col + ":N",
                        column=pbs_col + ":N",
                    )
                )
                st.altair_chart(cross_chart_pbs, use_container_width=True)
            else:
                st.info("No data for this cross analysis.")
        else:
            st.info("PBS column not available for cross analysis.")

    # ---------------------- DATA TABLE ----------------------
    st.subheader("Filtered data table")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data (CSV)",
        data=csv,
        file_name="protection_evaluation_filtered.csv",
        mime="text/csv",
    )

# --------------------------------------------------------
# PAGE 2 – INTERVIEW FORM (BASED ON EXISTING QUESTIONS)
# --------------------------------------------------------
else:
    st.title("Interview form – protection indicators")

    st.write(
        "This simple form mirrors the main structure of your Kobo questionnaire. "
        "It is mainly for learning and demonstration, not for production data collection."
    )

    if "new_responses" not in st.session_state:
        st.session_state["new_responses"] = []

    with st.form("protection_form"):
        st.subheader("Basic information")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            name_val = st.text_input("Name or code of respondent", "")
        with col_b:
            if camp_col is not None:
                camps = sorted(data[camp_col].dropna().unique())
                camp_val = st.selectbox("Camp", camps)
            else:
                camp_val = st.text_input("Camp", "")
        with col_c:
            sex_val = st.selectbox(
                "Sex",
                ["Féminin", "Masculin", "Other or prefer not to say"],
            )

        col_d, col_e = st.columns(2)
        with col_d:
            age_val = st.number_input(
                "Age", min_value=15, max_value=120, value=25, step=1
            )
        with col_e:
            pbs_val = st.selectbox(
                "Person with disability (self reported)",
                ["Oui", "Non", "Ne sait pas"],
            )

        st.markdown("---")
        st.subheader("Key indicator questions")

        responses = {}
        if not indicator_cols:
            st.warning("No indicator columns detected. Form part is limited.")
        else:
            # For each indicator column, use a radio with options taken from existing answers
            for col in indicator_cols:
                opts = sorted(data[col].dropna().astype(str).unique())
                # If too many options, treat as free text rather than radio
                if len(opts) <= 10:
                    responses[col] = st.radio(short(col), opts, horizontal=False)
                else:
                    responses[col] = st.text_input(short(col))

        comments_val = st.text_area(
            "Additional comments or protection concerns (optional)"
        )

        submitted = st.form_submit_button("Save interview in this session")

        if submitted:
            new_record = {
                "submission_time": datetime.utcnow().isoformat(),
                "Nom": name_val,
                camp_col if camp_col is not None else "Camp": camp_val,
                sex_col if sex_col is not None else "Sex": sex_val,
                age_col if age_col is not None else "Age": age_val,
                pbs_col if pbs_col is not None else "PBS": pbs_val,
                "comments": comments_val,
            }
            new_record.update(responses)
            st.session_state["new_responses"].append(new_record)
            st.success("Interview saved in the current session.")

    st.subheader("Interviews entered in this session")

    if len(st.session_state["new_responses"]) == 0:
        st.info("No interviews recorded yet in this session.")
    else:
        new_df = pd.DataFrame(st.session_state["new_responses"])
        st.dataframe(new_df, use_container_width=True)

        csv_new = new_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download new interviews (CSV)",
            data=csv_new,
            file_name="new_protection_interviews.csv",
            mime="text/csv",
        )

        st.caption(
            "Note: these new interviews are not merged automatically with your original Kobo export. "
            "You can download them and merge manually in Excel or another tool."
        )
