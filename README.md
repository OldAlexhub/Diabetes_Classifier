# ML Pipeline

This repository will host an end-to-end machine learning pipeline project.  
Currently, it contains environment setup instructions for Python 3.12 using Conda.

---

## 📦 Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/OldAlexhub/Diabetes_Classifier.git
cd ml-pipeline
```

### 2. Create a Conda Environment

```bash
conda create -n venv python=3.12
conda activate venv
```

## 📜 Overview

This project implements a complete ML workflow:

- Data ingestion (raw to processed)

- Exploratory data analysis in notebooks

- Feature engineering & preprocessing

- Model training & evaluation

- Deployment-ready artifacts for inference

Currently, the repo contains environment setup instructions for Python 3.12 using Conda, along with a scaffolded project structure.

## 🛠 Install Dependencies

Once your environment is activated, install required dependencies:

```bash
pip install -r requirements.txt
```

(Add a requirements.txt file listing your project dependencies.)

## 📌 Notes

- Use Python 3.12 for maximum compatibility.

- Avoid spaces in directory paths when creating environments.

- Keep raw data immutable — always store processed outputs in data/processed/.

- This repo is under active development — update the README as the project evolves.

## 🧑‍💻 Author

Mohamed Gad
www.mohamedgad.com
