# NearMiss

NearMiss is a a deep learning system that predicts satellite collision risks by combining orbital mechanics data with ML-based forecasting of orbital uncertainties. It uses Two-Line Element (TLE) data and SGP4 propagation to determine the closest approach and collision probabilities between satellites.

Project conceptually flows as follows.
1. TLE data for satellites is reterived from Celestrak API.
2. In order to make a neural network model, we need to create a training dataset and it is created by physically propagating the satellite pairs (which have potential of collision) using the physical algorithm made. For such satellite pairs, we extract their minimum distance of approach and the maximum probability of collision at this minimum distance.
3. We have used 3 stage neural network model in order to train the whole pipeline. The stages are **Filter**, **Approach**, and **Likelihood**.
4. In the *Filter* stage, our model filters out the pairs which are worthy enough to collide in the given time frame. The further processing happens on those pairs who pass the filter.
5. In the *Approach* stage, our model learns how to find minimum distance of approach using the attributes of satellite pair extracted from their raw TLE data.
6. In the *Likelihood* stage, our model learns how to find the maximum probability of collision at the minimum distance. We use the previously predicted minimum distance in this stage in order to connect previous stages with current stage.
7. After training the model, we can test it using different test set.
8. We can make the *fast*, *preliminary* predictions using the trained model and filter out the potential satellite pair candidates. Using these candidates, we can easily propagate them using heavy algorithms in order to get the actual collision scenerio on the desk.

---

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
  - [Data Creation](#data-creation)
  - [Model Training](#model-training)
  - [Model Prediction and Testing](#model-prediction-and-testing)
- [Directory Structure](#directory-structure)
- [License](#license)

---

## Installation

To install and set up the project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Om-S-Yeole/NearMiss.git
   cd NearMiss
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the Project**:
   Install the project as a package for easier imports:
   ```bash
   pip install -e .
   ```

---

## Setup

### Data Directory
The project expects a specific directory structure for raw and processed data:
- `data/raw`: Contains raw TLE files.
- `data/processed`: Contains processed datasets.
- `data/to_predict`: Contains datasets for making predictions.
- `data/training_eval`: Contains training evaluation results.

If the files are ran correctly, then all of these directories will be created automatically.

### Configuration
The project uses a YAML configuration file for attributes. Ensure the file `nearmiss/data/configs/attributes.yaml` exists and is correctly configured.

---

## Usage

The project provides a Command-Line Interface (CLI) for various tasks. Below are the available commands:

### Data Creation

To fetch data from an API and create training datasets, use the `data_creation.py` script:

```bash
python src/cli/data_creation.py <D_start> <D_stop> [options]
```

#### Arguments:
- `<D_start>`: Start time of the data creation window (format: `YYYY-MM-DD HH:MM:SS`).
- `<D_stop>`: End time of the data creation window (format: `YYYY-MM-DD HH:MM:SS`).

#### Options:
- `--retrieve_from_api`: Fetch data from the API.
- `--from_latest_raw_data_file`: Use the latest raw data file for data creation.
- `--raw_data_file_name <name>`: Specify the raw data file name.
- `--t_interval <seconds>`: Time interval for orbit propagation (default: 900 seconds).
- `--r_threshold_KDtree <km>`: Distance threshold for KDTree (default: 12 km).
- `--make_features_only_data`: Generate a dataset with only features for predictions.
- `--r_obj_1 <meters>`: Radius of the primary satellite.
- `--r_obj_2 <meters>`: Radius of the secondary satellite.
- `--Dist <km>`: Threshold distance for the Apoapsis-Periapsis filter.

---

### Model Training

To train the full 3-stage neural network model, use the `train_model.py` script:

```bash
python src/cli/train_model.py [options]
```

#### Options:
- `--from_latest_processed_file`: Use the latest processed file (default: True).
- `--processed_file_name <name>`: Specify the processed file name.
- `--save_training_evaluations`: Save training evaluations to a file (default: True).
- `--filter_stage_lr <float>`: Learning rate for the filter stage.
- `--filter_stage_epochs <int>`: Number of epochs for the filter stage.
- `--approach_stage_lr <float>`: Learning rate for the approach stage.
- `--approach_stage_epochs <int>`: Number of epochs for the approach stage.
- `--likelihood_stage_lr <float>`: Learning rate for the likelihood stage.
- `--likelihood_stage_epochs <int>`: Number of epochs for the likelihood stage.
- `--filter_rej_code_threshold <float>`: Threshold for the filter rejection code (default: 0.5).

---

### Model Prediction and Testing

To test or make predictions using the pretrained model, use the `model_pred_test.py` script:

```bash
python src/cli/model_pred_test.py <file_name> <mode> [options]
```

#### Arguments:
- `<file_name>`: Name of the file containing input data.
- `<mode>`: Operation mode (`test` or `predict`).

#### Options:
- `--batch_size <int>`: Batch size for dataloaders (default: 512).

#### Note:
- For making predictions, files present in directory `data/to_predict/` will be used

---

## Directory Structure

The project follows this structure:

```
NearMiss/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── to_predict/
│   ├── training_eval/
│   └── configs/
│       └── attributes.yaml
├── src/
│   ├── cli/
│   │   ├── data_creation.py
│   │   ├── train_model.py
│   │   └── model_pred_test.py
│   ├── ml/
│   │   ├── data/
│   │   ├── models/
│   │   ├── utils/
│   │   └── evaluation/
│   └── nearmiss/
│       ├── astro/
│       └── data/
└── README.md
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.