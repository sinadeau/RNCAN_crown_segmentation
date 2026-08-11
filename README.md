# RNCAN Crown Segmentation

Deep learning pipeline for **tree crown segmentation from multispectral imagery** (RGB or RGB + NIR bands), developed as part of the GLO-7030 *Deep Learning* course project at Laval University (2025). The [TreeCountSegHeight](https://github.com/sizhuoli/TreeCountSegHeight) model was fined-tuned using additionnal data from 15,130 trees from 7 plantation sites in British Columbia, Canada. The species included in the training data are Western red cedar (\textit{Thuja plicata}, 10,196 trees) and Douglas-fir (\textit{Pseudotsuga menziesii} Mirbel, 4,934 trees). The resolution of multispectral images vary from 2,48 and 4,7 cm/pixel. For more details and results, see GLO_7030_FINAL_REPORT.pdf.

This repository builds upon the `TreeCountSegHeight` project and adapts it to the data, preprocessing workflow, configuration, fine-tuning, and inference requirements of this project.

The repository provides additional notebooks in folder `TreeCountSegHeight/files_added_in_the_repo`, including scripts to prepare multispectral imagery for fine-tuning and inference, and analyze the resulting predictions. The configuration files and other functions of the original `TreeCountSegHeight` code was also modified in a few places to adapt to our dataset (file paths, training hyperparameters, multispectral bands used).

> **Note:** The `TreeCountSegHeight` directory contains the upstream project and is not documented in detail here. Please refer to the [original repository](https://github.com/sizhuoli/TreeCountSegHeight) for its internal architecture and documentation.

---

## Project Workflow

The overall workflow is:

```text
Raw multispectral imagery
        │
        ▼
Data cleaning and normalization
        │
        ▼
Preprocessing and data extraction
        │
        ▼
Fine-tuning of pretrained model
        │
        ▼
Fine-tuned model
        │
        ▼
Inference on new imagery
        │
        ▼
Segmentation predictions
        │
        ▼
Metrics and visualization
```

---

## Repository Structure

```text
RNCAN_crown_segmentation/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── TreeCountSegHeight/
│   └── ... upstream TreeCountSegHeight project
│   └── files_added_in_the_repo:                                
│       ├── calculate_metrics.ipynb
│       ├── cleaning_raw_data.ipynb    
│       ├── preprocessing_for inference_and_viz.ipynb
│       ├── preprocessing_for inference_and_viz_RGB_NIR.ipynb 
│       └── split_train_test.ipynb
│       └── inference_vs_ground_truth_overlap_viz.ipynb
│
├── .gitignore
├── get_models.sh
├── GLO_7030_FINAL_REPORT.pdf
├── README.md
└── requirements.txt
```

### Main Components
Component       Description
TreeCountSegHeight/	Upstream TreeCountSegHeight project used as the deep learning framework
.devcontainer/	Development container configuration
get_models.sh	Script for obtaining the required pretrained models from TreeCountSegHeight/
requirements.txt	Python dependencies used by the additional project components

| Component | Description |
|---|---|
| `TreeCountSegHeight/` | Upstream TreeCountSegHeight project used as the deep learning framework |
| `.devcontainer/` | Development container configuration |
| `get_models.sh` | Script for obtaining the required pretrained models |
| `requirements.txt` | Python dependencies used by the additional project components |

### Additional project components

| Component | Description |
|---|---|
| `calculate_metrics.ipynb` | Calculation of evaluation metrics |
| `cleaning_raw_data.ipynb` | Remove artifacts and normalize the imagery to the expected 0–255 range |
| `preprocessing_for inference_and_viz.ipynb` |       Preparation and visualization of inference data for RGB images |
| `preprocessing_for inference_and_viz_RGB_NIR.ipynb` |       Preparation and visualization of inference data for RGB + NIR images |
| `split_train_test.ipynb` |  train/validation/test split |

---

## Requirements

The project requires:

- **Conda**
- **Python 3.10**
- **CUDA Toolkit 11.2**
- **cuDNN 8.1.0**
- **TensorFlow < 2.11**
- Dependencies listed in `requirements.txt`

The upstream `TreeCountSegHeight` project also provides its own environment configuration and requirements.

---

## Installation

Create and activate the Conda environment:

```bash
conda create -n env python=3.10
conda activate env
```

Install the required CUDA and cuDNN versions:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Install TensorFlow:

```bash
pip install "tensorflow<2.11"
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

### 1. Clean the Raw Imagery

Place the raw TIFF imagery to be processed in:

```text
TreeCountSegHeight/files_added_in_the_repo/preprocessing/raw_data/
```

The `cleaning_raw_data.ipynb` notebook is used to remove artifacts and normalize the imagery to the expected 0–255 range.

Set the input filename in the `image_name` variable and execute the notebook.

The cleaned image is then available in:

```text
TreeCountSegHeight/files_added_in_the_repo/preprocessing/cleaning_data/
```

### 2. Prepare Annotation Data

The Jupyter notebook TreeCountSegHeight/files_added_in_the_repo/split_train_test.ipynb can be used to randomly create training/validation/test rectangles.

Add the resulting training and validation shapefiles to the preprocessing directory:

```text
TreeCountSegHeight/files_added_in_the_repo/preprocessing/cleaning_data/
```

The preprocessing workflow expects the relevant files to follow the naming conventions used by `Preprocessing.py`, including:

- `example_annotation` for the annotation layer (tree crowns)
- `example_rectangle` for the training/validation/test rectangles or
- `example_rectangle_one_polygone` for the ground-truth configuration, where applicable

### 3. Run Preprocessing

From the `TreeCountSegHeight` directory, run:

```bash
python main0_preprocessing.py
```

The extracted data produced by the preprocessing stage is then available in:

```text
TreeCountSegHeight/files_added_in_the_repo/preprocessing/extracted_data/
```

---

## Fine-Tuning

The fine-tuning workflow uses the extracted data generated during preprocessing.

Copy the appropriate training and validation data to:

```text
TreeCountSegHeight/files_added_in_the_repo/fine_tuning/extracted_centroids_kernel5/
```

> **Important:** Only the required training and validation data should be used for the fine-tuning stage.

Run:

```bash
python main1-2_segcount_transfer_learning.py
```

Training parameters such as the number of epochs can be adjusted in:

```text
TreeCountSegHeight/config/UNetTrainingFinetune.py
```

The resulting fine-tuned model is saved under:

```text
TreeCountSegHeight/files_added_in_the_repo/fine_tuning/fine_tune_model/
```

---

## Inference

Inference is performed using the fine-tuned model.

### 1. Configure the Model

Update the paths in:

```text
TreeCountSegHeight/config/hyperps_local.yaml
```

In particular, ensure that `model_save_path` points to the fine-tuned model.

### 2. Prepare the Imagery

The original multispectral TIFF imagery contains **10 bands**, while the inference workflow uses either **3 or 4 bands**.

The following notebooks can be used to select the required channels, prepare the region of interest, and generate the input imagery for inference:

- To use RGB bands: `preprocessing_for_inference_and_viz.ipynb`
- To use RGB + NIR bands: `preprocessing_for_inference_and_viz_RGB_NIR.ipynb`

The resulting inference input should be placed in:

```text
TreeCountSegHeight/inference/inputs/
```

### 3. Run Inference

From the TreeCountSegHeight directory:

```bash
python main_local.py
```

The generated predictions are saved under:

```text
TreeCountSegHeight/inference/predictions/
```

---

## Evaluation and Visualization

The following Jupyter notebooks are provided for evaluating and visualizing the results:

- `TreeCountSegHeight/files_added_in_the_repo/calculate_metrics.ipynb` – calculation of evaluation metrics
- `TreeCountSegHeight/files_added_in_the_repo/inference_vs_ground_truth_overlap_viz.ipynb` – visualization the overlap between predictions and ground truth

---

## Example Data

The original `TreeCountSegHeight` repository includes example datasets to facilitate testing and understanding of the workflow.

---

## Models

The `get_models.sh` script is also provided to facilitate obtaining the required pretrained models from the original `TreeCountSegHeight` repository.

---

## Development Environment

A development container configuration is provided in:

```text
.devcontainer/devcontainer.json
```
---

This can be used to reproduce a consistent development environment when working with compatible container-based development tools.

## Acknowledgements

This project is based on **TreeCountSegHeight** by Sizhuo Li and collaborators.

The upstream project provides the core tree counting and crown segmentation framework on which this repository builds. Modifications were made to adapt the workflow to the project's specific data, preprocessing, configuration, training, and inference requirements.

Please consult the [upstream TreeCountSegHeight repository](https://github.com/sizhuoli/TreeCountSegHeight) for its original documentation, dependencies, and licensing information.

---

## License and Data Usage

This repository incorporates the TreeCountSegHeight project and therefore includes files subject to the licenses and usage conditions of the upstream project.

In addition, example datasets included in the repository may have their own data-use restrictions. Please consult the license and data-use documentation distributed with the corresponding files before redistributing or reusing them.

