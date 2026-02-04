# 🛒 Instacart Reorder Intelligence Engine

> **A data-driven system to predict and optimize product reorder behavior, driving customer retention and personalization strategy**

## 📊 Project Overview

This project analyzes **1.4M+ grocery orders** from Instacart to build a predictive system that identifies which products customers are likely to reorder. By understanding reorder patterns, we can:

- **Increase retention** through personalized recommendations
- **Optimize inventory** by predicting demand for high-reorder products
- **Improve UX** by surfacing frequently repurchased items
- **Drive revenue** through targeted promotions on habitual purchases

### 🎯 Business Impact

- **59.86% baseline reorder rate** across all products
- Identified top 500 "habit-forming" products with **85%+ reorder rates**
- Built ML model achieving **0.82 ROC-AUC** in predicting reorders
- Segmented products into 5 strategic categories for targeted marketing

## 🚀 Key Features

### 1. **Product Performance Analytics**
- Multi-dimensional product analysis (sales, reorders, basket position)
- Department and aisle-level insights
- Temporal patterns and trends

### 2. **Reorder Prediction Model**
- XGBoost classifier with advanced feature engineering
- Features: product popularity, reorder history, basket behavior
- Production-ready pipeline with model serialization

### 3. **Market Basket Analysis**
- Association rules mining (Apriori algorithm)
- Product affinity networks
- Cross-sell opportunity identification

### 4. **Customer Segmentation**
- Product clustering based on purchase behavior
- RFM-style analysis at product level
- Strategic product portfolio recommendations

### 5. **Interactive Dashboard**
- Real-time analytics and insights
- Model performance monitoring
- Business recommendations visualized

## 📁 Project Structure

```
instacart-reorder-intelligence/
│
├── src/
│   ├── data_processing.py      # ETL and feature engineering
│   ├── model_training.py       # ML model training pipeline
│   ├── analytics.py            # Business analytics functions
│   └── utils.py                # Helper utilities
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_development.ipynb
│
├── data/
│   └── processed/              # Processed datasets
│
├── models/
│   └── reorder_model.pkl       # Trained model
│
├── outputs/
│   ├── figures/                # Generated plots
│   └── reports/                # Analysis reports
│
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/instacart-reorder-intelligence.git
cd instacart-reorder-intelligence
```

### Step 2: Create Virtual Environment (Windows)
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Data
1. Download the Instacart dataset from [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis)
2. Place the following files in the `data/` directory:
   - `order_products__train.csv`
   - `products.csv`
   - `aisles.csv`
   - `departments.csv`

## 💻 Usage

### Run the Complete Pipeline
```bash
# Process data and train models
python src/data_processing.py
python src/model_training.py
python src/analytics.py
```

### Launch the Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📈 Methodology

### Data Processing
1. **Data Integration**: Merged orders, products, aisles, and departments
2. **Feature Engineering**: 
   - Product-level: reorder rate, total orders, avg basket position
   - Temporal: order sequence patterns
   - Categorical: department, aisle encodings
3. **Data Quality**: Handled missing values, outliers, and class imbalance

### Model Development
1. **Baseline Models**: Logistic Regression, Random Forest
2. **Advanced Models**: XGBoost, LightGBM
3. **Hyperparameter Tuning**: GridSearchCV with cross-validation
4. **Evaluation Metrics**: 
   - ROC-AUC (primary metric for ranking)
   - Precision-Recall (for business impact)
   - Feature importance analysis

### Analytics Approach
- **Product Segmentation**: K-Means clustering on behavioral features
- **Market Basket**: Apriori algorithm for association rules
- **Business Metrics**: Reorder rate, penetration, basket share

## 📊 Key Findings

### Product Insights
1. **Top Reorder Categories**: Organic produce, dairy, and personal care
2. **Basket Behavior**: First-added products have **23% higher reorder rates**
3. **Sweet Spot**: Products with 50-500 total orders show optimal reorder engagement

### Model Performance
- **XGBoost Model**: 0.82 ROC-AUC, 0.76 Precision at 30% recall
- **Key Predictors**:
  1. Historical reorder rate (feature importance: 0.42)
  2. Product popularity (0.28)
  3. Average basket position (0.15)

### Business Recommendations
1. **Personalization**: Surface high-reorder products in "Buy Again" section
2. **Inventory**: Stock 500 high-reorder items with 2x buffer
3. **Marketing**: Target customers with reorder-trigger campaigns at day 7-10
4. **Product Dev**: Develop store-brand alternatives for top reorder items


## 🧪 Testing
```bash
pytest tests/
```

## 📚 Technical Stack

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Development**: Jupyter, Git

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **Product Analytics**: KPI definition, funnel analysis, cohort behavior
- **Machine Learning**: Classification, feature engineering, model evaluation
- **Business Acumen**: Translating data insights into actionable recommendations
- **Data Engineering**: ETL pipelines, data quality, scalable processing
- **Communication**: Dashboard design, storytelling with data

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.


## 🙏 Acknowledgments

- Dataset: Instacart via Kaggle 
- Inspired by real-world product data science challenges at leading tech companies

---

**⭐ If you found this project helpful, please give it a star!**
