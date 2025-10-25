# Topic Modeling on News Articles

This notebook explores topic discovery on a BBC news dataset using classical topic modelling methods (LDA and NMF). The notebook contains end-to-end steps: data loading, preprocessing (tokenization, stopword removal, POS-aware lemmatization), building dictionary/corpus, training topic models, measuring coherence, and visualizing topics with pyLDAvis.

## Requirements (recommended)
- Python 3.8+  
- pandas, numpy, scikit-learn, gensim, pyLDAvis
- nltk, num2words, contractions  

## Installation (Windows PowerShell)
Create & activate virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
Install packages:
```powershell
pip install -r requirements.txt
```
Run NLTK downloads in the notebook (punkt, stopwords, wordnet, averaged_perceptron_tagger).

## Usage
1. Open `Topic Modeling on News Articles.ipynb` in VS Code or Jupyter and run cells sequentially.  

## Acknowledgments
- The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/gpreda/bbc-news).
