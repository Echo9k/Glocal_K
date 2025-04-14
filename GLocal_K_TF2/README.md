# GLocal_K_TF2 Project

## Overview
GLocal_K_TF2 is a TensorFlow 2 project designed for collaborative filtering and recommendation systems. It implements a model that utilizes local and global kernel layers to predict user ratings for items based on historical data.

## Project Structure
```
GLocal_K_TF2
├── src
│   ├── data_loader.py      # Functions for loading datasets
│   ├── evaluate.py         # Evaluation functions for model performance
│   ├── main.py             # Entry point for the application
│   └── model.py            # Model architecture and training functions
└── README.md               # Project documentation
```

## Files Description

### `src/data_loader.py`
This file contains functions for loading datasets:
- `load_data_100k`: Loads data from the MovieLens 100K dataset.
- `load_data_1m`: Loads data from the MovieLens 1M dataset.
- `load_data_monti`: Loads data from the Douban dataset.
- `load_matlab_file`: Loads MATLAB files for data processing.

### `src/evaluate.py`
This file includes evaluation functions:
- `dcg_k`: Computes the discounted cumulative gain.
- `ndcg_k`: Computes the normalized discounted cumulative gain.
- `call_ndcg`: Calculates the NDCG score for predictions against true labels.

### `src/main.py`
The main entry point for the application:
- Sets up the data path and selects the dataset.
- Loads the data using functions from `data_loader.py`.
- Initializes the model and contains the training and evaluation loop.

### `src/model.py`
Defines the model architecture:
- `LocalKernelLayer`: A custom layer for local kernel operations.
- `GlobalKernelLayer`: A custom layer for global kernel operations.
- Functions for creating, training, and fine-tuning the model.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd GLocal_K_TF2
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your dataset and update the data path in `main.py`.

## Usage
To run the application, execute the following command:
```
python src/main.py
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.