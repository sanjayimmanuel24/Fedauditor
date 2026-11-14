# Notebooks

`patients.ipynb` is the exploratory notebook that preceded the federated
engine. It is kept for provenance and for readers who want to see the original
PIMA workflow.

The current application does **not** import the notebook or its artifacts. The
production path is:

- `data_loader.py` for loading and partitioning
- `federated_engine.py` for the FedAvg loop
- `app.py` for the Flask dashboard
- `experiments.py` for reproducible multi-seed results

## `artifacts/`

The files in `notebooks/artifacts/` were produced by the notebook:

- `global_model.pkl`
- `scaler.pkl`
- `global_weights.npz`
- `sample_patients.csv`
- `library_versions.json`

They are legacy reference artifacts, not inputs to the current engine. See
`artifacts/README.md` for details.
