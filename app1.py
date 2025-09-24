import streamlit as st
import io  # for download functionality
from helper import (
    file_path, overview, summary_stats, missing_values, outliers_check,
    hist_plot, barplot, correlation, box_plot, preprocessing, Ouliers, save,
    llm_insights
)

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="ML Ready AI", layout="wide")

st.title("🤖 ML Ready AI")
st.caption("Your Automated Data Scientist – Upload, Explore, Clean & Get ML Insights")


# --------------------------
# Sidebar Navigation
# --------------------------
st.sidebar.header("Navigation")
pages = [
    "Upload Data",
    "Overview",
    "EDA - Plots & Stats",
    "Preprocessing",
    "Outlier Removal",
    "LLM Insights",
    "Save Data"
]
choice = st.sidebar.radio("Choose a step:", pages)

# --------------------------
# Initialize session state
# --------------------------
if "df" not in st.session_state:
    st.session_state.df = None

# --------------------------
# 1️⃣ Upload Data
# --------------------------
if choice == "Upload Data":
    st.subheader("📂 Upload your dataset")
    file = st.file_uploader("Upload CSV/Excel/JSON/XML/HTML", type=["csv", "xlsx", "json", "xml", "html"])
    
    if file is not None:
        file_name = file.name
        with open(file_name, "wb") as f:
            f.write(file.getbuffer())
        df = file_path(file_name)
        st.session_state.df = df
        st.success("✅ File uploaded successfully!")
    
    if st.session_state.df is not None:
        st.write("### Preview of Data")
        st.dataframe(st.session_state.df.head())

# --------------------------
# 2️⃣ Overview
# --------------------------
elif choice == "Overview":
    if st.session_state.df is not None:
        st.subheader("🔍 Dataset Overview")
        info = overview(st.session_state.df)
        st.write("**Shape:**", info["shape"])
        st.write("**Columns:**", info["columns"])
        st.write("**Data Types:**", info["dtypes"])
        st.write("**First 5 Rows:**")
        st.dataframe(info["head"])
    else:
        st.warning("⚠️ Please upload a dataset first.")

# --------------------------
# 3️⃣ EDA
# --------------------------
elif choice == "EDA - Plots & Stats":
    if st.session_state.df is not None:
        st.subheader("📊 Exploratory Data Analysis")
        
        st.write("### Summary Stats")
        stats = summary_stats(st.session_state.df)
        st.write("**Numeric Features**")
        st.dataframe(stats["numeric"])
        st.write("**Categorical Features**")
        st.dataframe(stats["categorical"])
        
        st.write("### Missing Values")
        st.write(missing_values(st.session_state.df))
        
        st.write("### Histograms")
        for fig in hist_plot(st.session_state.df):
            st.pyplot(fig)
        
        st.write("### Bar Plots")
        for fig in barplot(st.session_state.df):
            st.pyplot(fig)
        
        st.write("### Correlation Heatmap")
        st.pyplot(correlation(st.session_state.df))
        
        st.write("### Box Plots")
        for fig in box_plot(st.session_state.df):
            st.pyplot(fig)
    else:
        st.warning("⚠️ Please upload a dataset first.")

# --------------------------
# 4️⃣ Preprocessing
# --------------------------
elif choice == "Preprocessing":
    if st.session_state.df is not None:
        st.subheader("⚙️ Data Preprocessing")
        drop_cols = st.multiselect("Select columns to drop", st.session_state.df.columns.tolist())
        if st.button("Apply Preprocessing"):
            df_cleaned = preprocessing(st.session_state.df.copy(), drop_cols)
            st.session_state.df = df_cleaned
            st.success("✅ Preprocessing applied!")
            st.dataframe(df_cleaned.head())
    else:
        st.warning("⚠️ Please upload a dataset first.")

# --------------------------
# 5️⃣ Outlier Removal
# --------------------------
elif choice == "Outlier Removal":
    if st.session_state.df is not None:
        st.subheader("🚫 Outlier Removal using IQR")
        if st.button("Remove Outliers"):
            df_no_outliers = Ouliers(st.session_state.df.copy())
            st.session_state.df = df_no_outliers
            st.success("✅ Outliers removed!")
            st.dataframe(df_no_outliers.head())
    else:
        st.warning("⚠️ Please upload a dataset first.")

# --------------------------
# 6️⃣ LLM Insights
# --------------------------
elif choice == "LLM Insights":
    if st.session_state.df is not None:
        st.subheader("🤖 AI Insights on Dataset")
        api_key = st.text_input("Enter your Groq API Key", type="password")
        if st.button("Generate Insights") and api_key:
            with st.spinner("Generating insights..."):
                response = llm_insights(st.session_state.df, api_key)
            st.markdown(response)
    else:
        st.warning("⚠️ Please upload a dataset first.")

# --------------------------
# 7️⃣ Save Data with Download
# --------------------------
elif choice == "Save Data":
    if st.session_state.df is not None:
        st.subheader("💾 Save Processed Data")

        # Choose format
        format_choice = st.selectbox("Choose format", ["csv", "excel", "json", "html"])

        if st.button("Save File"):
            # Save to disk
            path = save(st.session_state.df, format_choice)
            st.success(f"✅ File saved successfully as {path}")

            # --------------------------
            # Provide download option
            # --------------------------
            if format_choice.lower() == "csv":
                buffer = io.StringIO()
                st.session_state.df.to_csv(buffer, index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=buffer.getvalue(),
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )

            elif format_choice.lower() in ["excel", "xlsx"]:
                buffer = io.BytesIO()
                st.session_state.df.to_excel(buffer, index=False, engine='xlsxwriter')
                st.download_button(
                    label="⬇️ Download Excel",
                    data=buffer.getvalue(),
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            elif format_choice.lower() == "json":
                buffer = io.StringIO()
                st.session_state.df.to_json(buffer, orient="records", lines=True)
                st.download_button(
                    label="⬇️ Download JSON",
                    data=buffer.getvalue(),
                    file_name="cleaned_data.json",
                    mime="application/json"
                )

            elif format_choice.lower() == "html":
                buffer = io.StringIO()
                st.session_state.df.to_html(buffer, index=False)
                st.download_button(
                    label="⬇️ Download HTML",
                    data=buffer.getvalue(),
                    file_name="cleaned_data.html",
                    mime="text/html"
                )
    else:
        st.warning("⚠️ Please upload a dataset first.")
