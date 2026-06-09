import streamlit as st
import subprocess
import pandas as pd
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Agentic AutoML",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("Agentic AutoML System")
st.write("Multi-Agent LLM Powered AutoML Pipeline")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Configuration")

# ---------------------------------
# DATA INPUT MODE
# ---------------------------------

data_source = st.sidebar.radio(
    "Choose Dataset Source",
    [
        "Upload CSV",
        "Dataset URL"
    ]
)

uploaded_file = None
dataset_url = None

# ---------------------------------
# FILE UPLOAD
# ---------------------------------

if data_source == "Upload CSV":

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

# ---------------------------------
# URL INPUT
# ---------------------------------

else:

    dataset_url = st.sidebar.text_input(
        "Dataset CSV URL"
    )

# ---------------------------------
# TARGET COLUMN
# ---------------------------------

target_column = st.sidebar.text_input(
    "Target Column"
)

# ---------------------------------
# ITERATIONS
# ---------------------------------

iterations = st.sidebar.slider(
    "Max Iterations",
    1,
    10,
    3
)

# ---------------------------------
# LLM MODEL
# ---------------------------------

llm_model = st.sidebar.selectbox(
    "LLM Model",
    [
        "phi3:mini",
        "mistral",
        "llama3"
    ]
)

# ---------------------------------
# RUN BUTTON
# ---------------------------------

run_button = st.sidebar.button(
    "Run Pipeline"
)

# =====================================================
# HANDLE DATA INPUT
# =====================================================

file_path = None

# ---------------------------------
# HANDLE FILE UPLOAD
# ---------------------------------

if uploaded_file:

    os.makedirs("temp", exist_ok=True)

    file_path = os.path.join(
        "temp",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:

        f.write(uploaded_file.getbuffer())

    st.sidebar.success(
        f"Uploaded: {uploaded_file.name}"
    )

# ---------------------------------
# HANDLE URL
# ---------------------------------

elif dataset_url:

    file_path = dataset_url

    st.sidebar.success(
        "Dataset URL Added"
    )

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Metrics",
    "Workflow",
    "Logs"
])

# =====================================================
# OVERVIEW TAB
# =====================================================

with tab1:

    st.subheader("System Overview")

    st.markdown("""
    This system performs:

    - Dataset profiling
    - LLM-based strategy generation
    - Pipeline generation
    - Model execution
    - Evaluation
    - Failure analysis

    using multiple autonomous agents.
    """)

    # -----------------------------------------
    # RUN PIPELINE
    # -----------------------------------------

    if run_button:

        # ---------------------------------
        # VALIDATION
        # ---------------------------------

        if not file_path:

            st.error(
                "Please upload dataset or enter dataset URL"
            )

        elif not target_column:

            st.error(
                "Please enter target column"
            )

        else:

            st.info(
                "Running Agentic AutoML Pipeline..."
            )

            # ---------------------------------
            # COMMAND
            # ---------------------------------

            command = [
                "python",
                "run_experiments.py",
                "--file",
                file_path,
                "--target",
                target_column
            ]

            # =====================================================
            # LIVE LOG STREAMING
            # =====================================================

            log_placeholder = st.empty()

            logs = ""

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                bufsize=1
            )

            # ---------------------------------
            # STREAM LOGS LIVE
            # ---------------------------------

            for line in process.stdout:

                logs += line

                log_placeholder.code(logs)

            # ---------------------------------
            # WAIT FOR COMPLETION
            # ---------------------------------

            process.wait()

            # ---------------------------------
            # STORE LOGS
            # ---------------------------------

            st.session_state["logs"] = logs

            # ---------------------------------
            # SUCCESS / FAILURE
            # ---------------------------------

            if process.returncode == 0:

                st.success(
                    "Pipeline Execution Completed"
                )

            else:

                st.error(
                    "Pipeline Execution Failed"
                )

# =====================================================
# METRICS TAB
# =====================================================

with tab2:

    st.subheader("Final Summary")

    summary_path = "experiments/results/final_summary.csv"

    if os.path.exists(summary_path):

        df = pd.read_csv(summary_path)

        # -----------------------------------------
        # METRIC CARDS
        # -----------------------------------------

        best_accuracy = round(
            df["accuracy"].max(),
            4
        )

        best_f1 = round(
            df["f1"].max(),
            4
        )

        fastest_runtime = round(
            df["runtime"].min(),
            2
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Best Accuracy",
            best_accuracy
        )

        col2.metric(
            "Best F1 Score",
            best_f1
        )

        col3.metric(
            "Fastest Runtime",
            f"{fastest_runtime}s"
        )

        # -----------------------------------------
        # DATAFRAME
        # -----------------------------------------

        st.dataframe(df)

    # -----------------------------------------
    # PLOTS
    # -----------------------------------------

    st.subheader("Performance Plots")

    plots_dir = "experiments/results/plots"

    if os.path.exists(plots_dir):

        cols = st.columns(3)

        plots = [
            "accuracy.png",
            "f1.png",
            "runtime.png"
        ]

        for i, plot in enumerate(plots):

            plot_path = os.path.join(
                plots_dir,
                plot
            )

            if os.path.exists(plot_path):

                cols[i].image(
                    plot_path,
                    caption=plot
                )

# =====================================================
# WORKFLOW TAB
# =====================================================

with tab3:

    st.subheader("Multi-Agent Workflow")

    st.markdown("""
    ```text
    Dataset Analyzer Agent
              ↓
    Strategy Agent
              ↓
    Pipeline Generation Agent
              ↓
    Execution Engine
              ↓
    Evaluation Agent
              ↓
    Failure Analysis Agent
    ```
    """)

    st.info("""
    Each agent performs a specialized task
    in the AutoML decision-making pipeline.
    """)

# =====================================================
# LOGS TAB
# =====================================================

with tab4:

    st.subheader("Execution Logs")

    if "logs" in st.session_state:

        st.code(
            st.session_state["logs"]
        )

    else:

        st.warning(
            "No logs available yet"
        )