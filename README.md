# 📊 Automated EDA with LLM

An AI-powered Exploratory Data Analysis (EDA) system that combines traditional data analytics with Large Language Models (LLMs) to provide intelligent dataset understanding, automated visualizations, conversational analytics, and smart preprocessing recommendations.

---

# 🚀 Project Overview

This project allows users to:

✅ Upload CSV datasets  
✅ Perform automated data profiling  
✅ Generate interactive visualizations  
✅ Detect missing values and outliers  
✅ Generate professional EDA reports  
✅ Get AI-powered dataset insights  
✅ Chat with datasets using natural language  
✅ Receive smart preprocessing and ML recommendations  

The system is designed as an M.Tech research project focused on intelligent dataset understanding using LLMs.

---

# 🧠 Project Title

### “LLM-Augmented Automated Exploratory Data Analysis System for Intelligent Dataset Understanding”

---

# 🏗️ System Architecture

```text
Frontend Layer (Streamlit)
        ↓
Data Processing Layer (Pandas)
        ↓
Visualization Layer (Plotly)
        ↓
Profiling Engine (ydata-profiling)
        ↓
Analytics Layer
        ↓
LLM Engine (Groq API)
        ↓
Insight Generation & Conversational Analytics
````

---

# ⚙️ Technologies Used

| Technology      | Purpose                    |
| --------------- | -------------------------- |
| Python          | Core Programming           |
| Streamlit       | Frontend UI                |
| Pandas          | Data Processing            |
| Plotly          | Interactive Visualizations |
| ydata-profiling | Automated EDA Reports      |
| Groq API        | LLM Integration            |
| NumPy           | Numerical Operations       |
| SciPy           | Statistical Analysis       |
| Statsmodels     | VIF Calculation            |
| Matplotlib      | Additional Plotting        |

---

# 📁 Project Structure

```text
automated-eda-llm/
│
├── app.py
├── requirements.txt
├── README.md
│
├── assets/
│   └── style.css
│
├── outputs/
│   └── eda_report.html
│
├── utils/
│   ├── profiling.py
│   ├── visualization.py
│   ├── profiling_report.py
│   ├── llm_analysis.py
│   ├── analytics.py
│
└── datasets/
```

---

# ✨ Features

## 📊 Automated Data Profiling

* Dataset overview
* Column analysis
* Data type detection
* Missing value analysis
* Summary statistics

---

## 📈 Interactive Visualizations

* Histograms
* Boxplots
* Correlation heatmaps
* Countplots
* Interactive Plotly charts

---

## 🧠 AI-Powered Insights

Using Groq LLM API:

* Dataset quality analysis
* Correlation insights
* Outlier analysis
* Smart preprocessing suggestions
* ML recommendations

---

## 💬 Conversational Analytics

Users can ask natural language questions such as:

* Which columns have missing values?
* Which feature is highly correlated?
* Are there outliers?
* What preprocessing is recommended?

---

## 🚨 Smart Analytics

Advanced analytics include:

* IQR Outlier Detection
* Z-Score Analysis
* Skewness Detection
* Duplicate Detection
* Multicollinearity Detection (VIF)

---

## 📄 Automated EDA Report

Generate:

* Full HTML profiling report
* Correlation analysis
* Missing value analysis
* Outlier reports
* Downloadable reports

---

# 🔧 Installation

## 1. Clone Repository

```bash
git clone <repository-url>
cd automated-eda-llm
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

# 🔑 Configure Groq API

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

Get API Key from:

[xAI Console](https://console.x.ai?utm_source=chatgpt.com)

or

[Groq Cloud](https://console.groq.com?utm_source=chatgpt.com)

---

# ▶️ Run Application

```bash
streamlit run app.py
```

---

# 📸 Application Modules

| Module               | Description                         |
| -------------------- | ----------------------------------- |
| Data Profiling       | Dataset structure and statistics    |
| Visualization        | Interactive chart generation        |
| Correlation Analysis | Feature relationship analysis       |
| AI Insights          | LLM-generated dataset understanding |
| Chat with Dataset    | Conversational analytics            |
| Smart Analytics      | Outlier & preprocessing analysis    |
| ML Recommendations   | Beginner-friendly ML guidance       |

---

# 📚 Research Contributions

This project contributes toward:

* Intelligent dataset understanding
* AI-assisted EDA workflows
* Conversational data analytics
* Automated preprocessing recommendations
* LLM integration in data science workflows

---

# 🎯 Future Scope

Possible future enhancements:

* PDF export support
* Dataset comparison
* AutoML integration
* Real-time data streaming
* Dashboard customization
* Multi-file analysis
* Advanced conversational memory
* Cloud deployment

---

# ⚠️ Limitations

* Works primarily with CSV datasets
* Depends on external LLM APIs
* Large datasets may increase processing time
* AI responses depend on prompt quality
* Limited advanced ML capabilities

---

