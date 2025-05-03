# 📘 GeoThesis – Geothermal Potential Modeling

**Author:** Daniel Herrera  
**Date:** 03/05/2025

GeoThesis is the full analytical and modeling pipeline that supports the GeoWatt ZH application. It includes all Jupyter notebooks used throughout the research process for estimating shallow geothermal potential in the Canton of Zürich — from raw spatial data extraction to the final trained machine learning model.

This repository serves as a transparent, reproducible research companion to the deployed tool and thesis report.

---

## 1. Project Overview

The GeoThesis repository documents the full data science workflow applied in the master's thesis:

- 🗺️ **Geospatial Data Extraction**  
- 🧮 **Feature Engineering and Data Transformation**  
- 📈 **Exploratory Data Analysis and Visualizations**  
- 🤖 **Machine Learning Model Development**  
- 📊 **Performance Evaluation and Generalization Tests**

The project integrates open-source data from Zürich’s GIS platforms, borehole records, and national boundary data to build an explainable model for heat yield estimation.

---

## 2. Repository Structure

- 📁 `jupyter_notebooks/` – Core analysis workflow, structured by stage (extraction, transformation, modeling).
- 📁 `scripts/` – Python helper scripts for spatial processing and model loading.
- 📁 `models/` – Trained machine learning artifacts (e.g., XGBoost `.pkl` files).
- 📁 `assets/` – Visual materials for figures and export.
- 📄 `main.py` – Entry point for the deployed web application (optional link to `GeoWattZH` repo).
- 📄 `README.md` – This file.

---

## 3. Jupyter Notebook Breakdown

Below is a summary of the contents in the `jupyter_notebooks/` folder:

| Filename | Description |
|---------|-------------|
| `00_analysis_zh_sonden.ipynb` | Initial exploratory analysis of geothermal probes in Zürich. |
| `00_widget_map.ipynb` | Prototype of a map widget for interacting with spatial layers. |
| `01_extraction_swissboundaries.ipynb` | Imports and processes national boundary shapefiles. |
| `02_extraction_zh_erdwaermezonen.ipynb` | Extracts geothermal zoning information from Zürich GIS sources. |
| `03_extraction_zh_mastersonden.ipynb` | Extracts detailed records of approved and installed boreholes. |
| `04_transformation_zh_mastersonden.ipynb` | Cleans and engineers features from borehole data. |
| `05_analysis_zh_mastersonden.ipynb` | Detailed statistical analysis and visualization of probe characteristics. |
| `06_analysis_zh_density_potential.ipynb` | Constructs a density-based potential map using hexagon tiling. |
| `07_predictive_model.ipynb` | Trains and evaluates machine learning models (XGBoost, baseline, linear). |
| `08_test_lu_erdwaermenutzung.ipynb` | Cross-canton generalization tests using Luzern data. |

---

## 4. Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
