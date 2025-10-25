# Topic Modeling on News Articles (Task 5)

This notebook explores topic discovery on a BBC news dataset using classical topic modelling methods (LDA and NMF). The notebook contains end-to-end steps: data loading, preprocessing (tokenization, stopword removal, POS-aware lemmatization), building dictionary/corpus, training topic models, measuring coherence, and visualizing topics with pyLDAvis.

## Files
- `Task 5 Topic Modeling on News Articles.ipynb` — main notebook with data ingest, preprocessing, LDA/NMF, coherence checks and visualization.
- `requirements.txt` — Python packages needed to run the notebook.

## Quickstart (recommended)
1. Create and activate a virtual environment (example using venv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start Jupyter (Lab or Notebook) and open the notebook:

```powershell
jupyter lab
```

4. Run cells top-to-bottom. If the installation cell installs new packages, restart the kernel and re-run.

## Important notes
- NLTK data and spaCy model: the notebook contains `nltk.download(...)` calls and attempts to download `en_core_web_sm` for spaCy if missing. If you prefer, run these manually after installing packages:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import spacy
spacy.cli.download('en_core_web_sm')
```

- pyLDAvis requires the corpus to be a gensim bag-of-words corpus (list of (term_id, count) tuples). If you encounter ValueError during visualization, confirm the notebook built `gensim_corpus` as:

```python
gensim_corpus = [dictionary.doc2bow(text) for text in texts]
```

## Recommended next steps
- Sweep `num_topics` (e.g., 4–20) and plot coherence (c_v) to select the best topic count.
- Save the best LDA/NMF models and export document-topic assignments to CSV for downstream analysis.
- Consider experimenting with BERTopic or transformer-based clustering for improved semantic coherence.

## Contact
If you want, I can add helper cells to (1) grid-search topics and plot coherence vs topics, (2) export per-document dominant topic to CSV, or (3) persist/load models.
