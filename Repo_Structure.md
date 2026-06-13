oil-price-predictor/
├── .gitignore
├── README.md
├── requirements.txt
├── config.yaml
├── Dockerfile
├── data/                  # Raw + processed
├── notebooks/             # EDA, experiments
├── src/
│   ├── data/              # loaders, merging prices + production
│   ├── features/          # engineering, selection
│   ├── models/            # training, evaluation, base model
│   ├── llm/               # LLM prompts, analysis
│   ├── utils/
│   └── pipeline.py
├── app/
│   ├── app.py             # Streamlit interface
│   └── utils.py
├── models/                # Saved models
├── tests/
└── logs/