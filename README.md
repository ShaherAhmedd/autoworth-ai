# AutoWorth AI — Used Car Price & Deal Advisor

AutoWorth is a machine learning project that estimates the market price of a used car from its listing details, then compares that estimate with a seller's asking price. The Streamlit app also gives a quick look at the historical listings used to train the model.

> **Scope:** This is an educational estimate built on historical UK listings. It is not a formal valuation or a guarantee of a car's sale price. Condition, service history, location, and current market changes are not represented in the model.

## What it does

- Combines listings for Audi, BMW, Ford, Hyundai, Mercedes, Skoda, Toyota, Vauxhall, and Volkswagen.
- Removes exact duplicates and records with impossible years after 2021; retains unusual but plausible records such as electric cars with a zero engine size.
- Engineers `CarAge`, `MileagePerYear`, and `CarAgeSquared` features.
- Scales numerical features and one-hot encodes categories inside a scikit-learn pipeline.
- Fits a Random Forest regressor and reports held-out R², MAE, and RMSE.
- Compares an asking price with the estimate using the notebook's deal bands: great deal, good deal, fair price, slightly overpriced, and overpriced.
- Displays median price by year and make, plus a mileage-versus-price chart.

On the included files, the cleaning step leaves **97,711 listings**. The app's reproducible held-out test split currently reports R² **0.955**, MAE **£1,180**, and RMSE **£2,109**. These values are app-model results and can vary slightly across library versions.

The original exploratory and model-comparison workflow is included at [`notebooks/AutoWorth_AI_Simple_Complete_Project.ipynb`](notebooks/AutoWorth_AI_Simple_Complete_Project.ipynb).

## Project layout

```text
autoworth-ai/
├── app.py
├── requirements.txt
├── src/
│   └── autoworth.py
├── data/                 # Nine included manufacturer CSVs
├── notebooks/
│   └── AutoWorth_AI_Simple_Complete_Project.ipynb
└── .streamlit/
    └── config.toml
```

## Dataset setup

The notebook uses the [100,000 UK Used Car Data set](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes). The nine CSVs used by the notebook are included in `data/`; the dataset page currently lists its license as **CC0: Public Domain**. The MIT license in this repository applies to the project code, not to third-party source material.

The app expects these nine filenames:

```text
audi.csv  bmw.csv  ford.csv  hyundi.csv  merc.csv
skoda.csv  toyota.csv  vauxhall.csv  vw.csv
```

The included files are ready for local use and for a public Streamlit deployment. Alternatively, launch the app and upload a ZIP containing the nine CSVs, or select the files in the sidebar. Extra `focus.csv`, `cclass.csv`, and unclean files are not used, matching the notebook's workflow.

## Run locally on Windows

Open PowerShell in the project folder, then run these commands one at a time:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The CSVs are already in `data/`. Start the web app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your browser. If PowerShell says Python was not found, select the installed Python interpreter in VS Code or use the full path to `python.exe`.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload or push this project.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Choose **Create app**, then select the repository, `main` branch, and `app.py` as the app file.
4. Deploy. The repository's `requirements.txt` installs the app dependencies.
5. Copy the generated `*.streamlit.app` URL into your LinkedIn post.

See Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) for current steps. Choose a Python version supported by both Community Cloud and the packages; Python 3.12 is a practical deployment choice. The app's local virtual environment is not uploaded.

## Model notes

The notebook compares Linear Regression, Decision Tree, Random Forest, and Gradient Boosting, then tunes a Random Forest. The interactive app trains its own reproducible Random Forest pipeline for a responsive demo. Its test metrics are computed from the same historical data at app startup and may differ from the notebook's tuned-model results. The test set is held out from model fitting; it is not a promise of future prediction accuracy.

## LinkedIn post draft

> I built AutoWorth AI, a used-car price and deal advisor with Python, scikit-learn, and Streamlit. It combines UK listings across nine car brands, compares regression models in the notebook, and turns a car's details into an estimated market price and asking-price comparison. I also built an interactive dashboard to explore the dataset and inspect held-out model metrics.
>
> Try the app: [your Streamlit link]  
> Explore the code: [your GitHub link]
>
> This is an educational project using historical listings, so the estimate is not a formal vehicle valuation.

## License

The code in this repository is provided under the [MIT License](LICENSE). The dataset remains subject to its own terms; see the dataset page before sharing it.
