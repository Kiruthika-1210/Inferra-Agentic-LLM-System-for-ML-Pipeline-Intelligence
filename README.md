# 🚀 LLM-Driven Agentic AutoML System

An end-to-end **Agentic AutoML framework** that uses Large Language Models (LLMs) to automatically design, evaluate, and refine machine learning pipelines through an **iterative feedback loop**.

---

## 📌 Overview

This project simulates how a real ML engineer works:

> Understand data → Choose model → Train → Evaluate → Fix mistakes → Repeat

But here, the entire process is handled by an **LLM-powered agent system**.

---

## 🧠 Key Features

* 🤖 **LLM-driven strategy generation**
* 🔁 **Iterative optimization loop (self-improving system)**
* 📊 **Automatic dataset profiling & insights**
* ⚙️ **Dynamic pipeline generation**
* 🧪 **Execution + evaluation + failure analysis**
* 📈 **Benchmarking vs manual & grid search baselines**
* 🧾 **Full experiment logging & traceability**

---

## 🏗️ Architecture

```text
Dataset → Profiling → LLM Insights → Strategy Generation
        → Pipeline Generation → Execution → Metrics
        → Evaluation → Failure Analysis → Refinement Loop
        → Best Strategy Selection → Benchmark Comparison
```

---

## 🔄 Workflow

### 1️⃣ Dataset Profiling

* Detects:

  * Dataset size
  * Feature types
  * Missing values
  * Class distribution

---

### 2️⃣ Insight Generation (LLM)

Generates:

* Insights
* Risk factors

---

### 3️⃣ Strategy Generation (LLM)

Outputs:

* Model selection
* Preprocessing steps
* Hyperparameters
* Confidence score

---

### 4️⃣ Pipeline Generation

* Uses `ColumnTransformer`
* Handles:

  * Missing values (mandatory)
  * Categorical encoding
  * Scaling (LLM-guided)
  * SMOTE (if needed)

---

### 5️⃣ Execution Engine

* Trains model
* Generates predictions
* Tracks:

  * Runtime
  * Peak memory

---

### 6️⃣ Metrics Engine

* Accuracy (train/test)
* Precision / Recall / F1-score
* Runtime
* Peak memory
* Pipeline complexity

---

### 7️⃣ Evaluation Agent

Detects:

* Good fit
* Overfitting
* Underfitting
* Execution failure

---

### 8️⃣ Failure Analysis Agent

Suggests improvements:

* Reduce complexity
* Change model
* Adjust preprocessing

---

### 9️⃣ Iterative Loop

Stops when:

* Target accuracy reached ✅
* Max iterations reached (default: 3)
* Improvement < threshold (e.g., 0.5%)

---

## 📊 Experiment Framework

### 📂 Structure

```
experiments/
├── baselines.py
├── run_experiments.py
├── results/
│   └── <dataset_name>/
│       ├── manual.csv
│       ├── gridsearch.csv
│       ├── agentic.csv
│       ├── summary.csv
```

---

## ⚖️ Baselines

### 🔹 Manual Baseline

* Fixed pipeline
* RandomForest model
* No tuning

---

### 🔹 GridSearch Baseline

* Same pipeline
* Uses GridSearchCV

---

### 🔹 Agentic Pipeline

* LLM-driven
* Adaptive
* Iterative refinement

---

## 📈 Metrics Tracked

* Accuracy
* Precision
* Recall
* F1-score
* Runtime
* Peak memory usage
* Pipeline complexity:

  * Number of steps
  * Number of hyperparameters

---

## ▶️ How to Run

### Run Agentic System

```bash
python main.py --file <dataset_path_or_url> --target <target_column>
```

### Run Full Experiment (Baselines + Agent)

```bash
python run_experiments.py --file <dataset_path_or_url> --target <target_column>
```

---

## 🧪 Example

```bash
python run_experiments.py \
--file https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv \
--target species
```

---

## 🧾 Sample Output

* Accuracy: **0.9667**
* F1 Score: **0.9666**
* Iterations: **1 (early stopping)**
* Model: GradientBoosting

---

## 📁 Logs

Each run generates structured logs:

```
experiments/logs/<dataset_name>_experiment_log.json
```

Includes:

* Dataset profile
* Insights
* Iterations (strategy + metrics + evaluation)
* Final result

---

## 💡 Key Highlights

* 🔥 Combines **LLM reasoning + ML pipelines**
* 🔥 Fully **dataset-agnostic**
* 🔥 Implements **feedback-driven learning**
* 🔥 Provides **experiment-level comparison**
* 🔥 Tracks full **decision trace**

---

## 🚀 Future Improvements

* RAG-based preprocessing knowledge
* Feature selection agent
* Regression support
* Visualization dashboard
* Better LLM response parsing

---

## 🧠 Tech Stack

* Python
* Scikit-learn
* Imbalanced-learn
* Pandas / NumPy
* LLM APIs (for reasoning agents)

---

## 🏁 Conclusion

This project demonstrates how LLMs can evolve from text generation tools into **decision-making systems**, enabling intelligent, adaptive machine learning workflows.

---

## 👩‍💻 Author

**Kiruthika M (Kittu)**
AI & Data Science Student | Aspiring SDE 🚀

---
