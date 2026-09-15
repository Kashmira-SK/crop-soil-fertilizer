# Precision Fertilizer Recommendation

AI/ML group project for predicting the appropriate fertilizer based on soil conditions, crop type, climate conditions, and soil nutrient levels.

## Dataset

The dataset contains 8,000 records with the following input features:

- Temparature
- Humidity
- Moisture
- Soil Type
- Crop Type
- Nitrogen
- Potassium
- Phosphorous

Target variable:

- Fertilizer Name

The target contains 7 fertilizer classes.

## Individual Preprocessing Contributions

| Student ID | Technique |
|---|---|
| IT25102351 | Encoding Categorical Variables |
| IT25102258 | Feature Engineering |
| IT25102263 | Outlier Detection and Treatment |
| IT25102315 | Feature Selection / Redundancy Analysis |
| IT25102320 | Feature Scaling |
| IT25102347 | Target Encoding and Class Balance Analysis |
| IT25300301 | Data Validation and Stratified Train-Test Split |

## Repository Structure

- `data/raw/` - Original assigned dataset
- `data/external/` - External datasets if used
- `notebooks/` - Individual preprocessing notebooks
- `group_pipeline.ipynb` - Combined preprocessing pipeline
- `results/eda_visualizations/` - EDA plots
- `results/logs/` - Execution logs if required
- `results/outputs/` - Processed datasets or generated outputs

## Running the Project

1. Open the required notebook in Google Colab or Jupyter Notebook.
2. Ensure `data_core.csv` is available from `data/raw/`.
3. Run the notebook cells from top to bottom.

The individual notebooks demonstrate each member's preprocessing contribution. The combined preprocessing workflow is available in `group_pipeline.ipynb`.
