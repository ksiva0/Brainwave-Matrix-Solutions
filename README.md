# 📰 Fake News Detection using Machine Learning

This project builds a **Fake News Detection Model** using **Logistic Regression** with **TF-IDF** vectorization. The model classifies news articles as **Real or Fake** based on text analysis.

## 📌 Features
- Text preprocessing (cleaning, stopwords removal)
- TF-IDF vectorization for feature extraction
- Logistic Regression with **hyperparameter tuning**
- Model evaluation (accuracy, classification report, confusion matrix)
- Visualizations: **Word Clouds, Confusion Matrix, ROC Curve, Feature Importance**


## 🚀 Installation

Clone the repository and install dependencies:

git clone https://github.com/ksiva0/Fake-News-Detection.git
cd Fake-News-Detection
pip install -r requirements.txt

#### 📂 Dataset
Ensure you have the dataset (fake_and_real_news.csv) in the project folder. If not, modify the file path accordingly.

### 📊 Results & Visualizations
- Model Accuracy: ~99.6% (Logistic Regression with Tuning)
- Confusion Matrix: High precision & recall for Fake/Real labels
- ROC Curve: AUC Score close to 1.0
- Top Words: Key words influencing Fake/Real classification

### Visualization	Description
🖼️ Word Cloud	Most : common words in Fake vs. Real news
🔥 Confusion Matrix	 : Model performance in classifying news
📈 ROC Curve :	Evaluation of model performance
📝 Feature Importance	 : Key words contributing to classification

### 📌 Future Enhancements
- Implement Deep Learning models (LSTM, BERT)
- Deploy as a Web App using Flask or Streamlit
- Expand dataset for better generalization

#### 🤝 Contributing
Pull requests are welcome! Feel free to open an issue for improvements. 🚀

### 🛠 Technologies Used
- Python 🐍
- Pandas, NumPy
- Scikit-Learn
- Matplotlib, Seaborn
-WordCloud
