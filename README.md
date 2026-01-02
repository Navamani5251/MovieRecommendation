# 🎬 Movie Recommendation App

## 📌 Overview

The **Movie Recommendation App** is a machine learning–based web application that recommends movies similar to a user-selected title. It uses **Natural Language Processing (NLP)** techniques to analyze movie metadata and provides content-based recommendations in real time through an interactive web interface.

This project is designed as a **college-level / internship-ready ML project** and demonstrates the complete workflow from data preprocessing to model deployment.

---

## 🚀 Features

* Content-based movie recommendations
* Uses genres, keywords, and movie overviews
* NLP-based text preprocessing
* TF-IDF vectorization
* Cosine similarity for recommendation
* Interactive **Streamlit** web interface
* Efficient model loading using **joblib**

---

## 🧠 Technologies Used

* **Python**
* **Pandas** – Data handling
* **NLTK** – Text preprocessing
* **Scikit-learn** – TF-IDF & similarity computation
* **Joblib** – Model persistence
* **Streamlit** – Web application frontend
* **Git LFS** – Handling large files

---

## 📂 Project Structure

```
Movie-Recommendation-App/
│── app.py                # Streamlit application
│── preprocess.py         # Data preprocessing script
│── recommend.py          # Recommendation logic
│── movies.csv            # Dataset (tracked using Git LFS)
│── df_cleaned.pkl        # Preprocessed data (LFS)
│── tfidf_matrix.pkl      # TF-IDF matrix (LFS)
│── cosine_sim.pkl        # Similarity matrix (LFS)
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/movie-recommendation-app.git
cd movie-recommendation-app
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download NLTK Data (First Run Only)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## ▶️ How to Run the Application

### Step 1: (Optional) Preprocess Data

Run this only if `.pkl` files are not present:

```bash
python preprocess.py
```

### Step 2: Run Streamlit App

```bash
streamlit run app.py
```

Open your browser and go to:

```
http://localhost:8501
```

---

## 📊 How It Works

1. Movie metadata is cleaned using NLP techniques
2. Text is converted into numerical vectors using **TF-IDF**
3. **Cosine similarity** calculates similarity between movies
4. The most similar movies are recommended to the user

---

## 📈 Future Enhancements

* Add user-based or hybrid recommendation
* Include movie posters using TMDB API
* Deploy on Streamlit Cloud / Render
* Improve UI with filters and ratings

---

## 🎯 Use Cases

* Academic mini / major project
* ML & NLP portfolio project
* Interview-ready demonstration
* Recommendation system learning

---

## 📝 Resume Description

> Built a content-based movie recommendation system using NLP, TF-IDF, and cosine similarity, deployed as an interactive Streamlit web application.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is licensed for educational use.

---

## 👨‍💻 Author

**Navamani Kandan**
Undergraduate Student | Machine Learning Enthusiast

---

⭐ If you like this project, don’t forget to star the repository!
