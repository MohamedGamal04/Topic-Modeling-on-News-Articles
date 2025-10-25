# Topic Modeling on News Articles

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

