# Topic Modeling on News Articles

This notebook explores topic discovery on a BBC news dataset using classical topic modelling methods (LDA and NMF). The notebook contains end-to-end steps: data loading, preprocessing (tokenization, stopword removal, POS-aware lemmatization), building dictionary/corpus, training topic models, measuring coherence, and visualizing topics with Wordcloud.

Live demo : https://topic-modeling-on-news-articles-tc3vzqsj4u5npm4lhxcs3h.streamlit.app/
## Features
- Data cleaning and lemmatization using NLTK
- TF‑IDF feature extraction
- Model selection and hyperparameter tuning (LDA , NMF)
- Feed‑forward neural network with TF‑IDF input
- Word clouds and feature importance visualization
- Streamlit UI demo for interactive sentiment analysis

## Project Structure

- `model.pkl` `vectorizer.pkl` - model and vectorizer files
- `Topic Modeling on News Articles.ipynb` - Jupyter Notebook containing preprocessing, training, and evaluation pipeline
- `app.py` - Streamlit application script for live sentiment analysis
- `requirements.txt` - Required Python packages
- `README.md` - Project overview and usage instructions

## Installation

1. Clone the repository:
``` powershell
git clone https://github.com/MohamedGamal04/Topic-Modeling-on-News-Articles.git
cd Topic-Modeling-on-News-Articles
```
2. Create a Python environment and activate it (recommended):
``` powershell
conda create -n classify-env python=3.8
conda activate classify-env
```
3. Install dependencies:
``` powershell
pip install -r requirements.txt
```
4. Download necessary NLTK data resources:
``` python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```
   
Run required NLTK downloads in the notebook (cells include downloads for punkt, stopwords, wordnet, averaged_perceptron_tagger).

## Usage

- Train the model and evaluate using the Jupyter Notebook.
- To use the Streamlit app (once created and set up), run:
``` terminal
streamlit run app.py
```

- The Streamlit UI allows you to enter reviews manually or upload CSV files for batch prediction.

## Dataset

This project uses the [BBC news classification dataset](https://www.kaggle.com/datasets/gpreda/bbc-news)
