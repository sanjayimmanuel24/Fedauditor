# Contributing

## Development setup

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python seed_demo.py
python app.py
```

See [`docs/SETUP.md`](docs/SETUP.md) for the full guide.

## Before opening a pull request

1. Run the unit tests:

   ```powershell
   ./venv/Scripts/python.exe -m unittest discover -s tests -v
   ```

2. Run a quick import check:

   ```powershell
   ./venv/Scripts/python.exe -c "import app"
   ```

3. Confirm the app still starts and the dashboard loads.

4. Do not commit `users.db`, `.env`, `venv/`, `__pycache__/`, or `_smoke_*` output.

## Code style

- Follow the existing module layout and naming.
- Keep functions typed where the surrounding module already uses type hints.
- Add docstrings to new public functions and modules.
- Keep numerical logic in the measurement modules, not in Flask route handlers.
- Do not hard-code absolute paths.
- Do not add a dependency unless it is listed in `requirements.txt`.

## Adding a dataset

1. Register the dataset in `config.DATASETS`.
2. Add a normaliser in `data_loader.py` if the raw columns need mapping.
3. Add a provenance note in `data/README.md` and `docs/DATA.md`.
4. Add the CSV under `data/`.
5. Run both partition modes and the unit tests.

## Measurement changes

The project is a measurement study. If you change a metric, a calibration method, a fairness
rule, or an explanation diagnostic:

- State the estimand in the docstring.
- Report uncertainty where the existing code reports it.
- Update `docs/METHODS.md` and `docs/RESULTS.md` if the reported numbers change.
- Do not tune a threshold on the labels used for evaluation.

## Commit hygiene

- Keep commits focused.
- Explain the motivation in the commit message, not only the file changed.
- Never commit credentials, local databases, or generated smoke output.
- Generated results are tracked intentionally; regenerate them with `experiments.py` rather than
  editing CSV or JSON by hand.
