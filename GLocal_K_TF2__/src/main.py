# GLocal_K_TF2/src/main.py

import tensorflow as tf
from src.data_loader import load_data_100k, load_data_1m, load_data_monti
from src.model import build_model, train_model
from src.evaluate import call_ndcg

def main():
    # Set up the data path and dataset selection
    data_path = ''  # Specify your data path here
    dataset = 'ML-100K'  # Choose among 'ML-100K', 'ML-1M', 'Douban'

    # Load the dataset
    if dataset == 'ML-100K':
        n_m, n_u, train_r, train_m, test_r, test_m = load_data_100k(path=data_path + '/MovieLens_100K/')
    elif dataset == 'ML-1M':
        n_m, n_u, train_r, train_m, test_r, test_m = load_data_1m(path=data_path + '/MovieLens_1M/')
    elif dataset == 'Douban':
        n_m, n_u, train_r, train_m, test_r, test_m = load_data_monti(path=data_path + '/Douban_monti/')
    else:
        raise ValueError("Invalid dataset selection")

    # Build and compile the model
    model = build_model(n_m, n_u)

    # Train the model
    train_model(model, train_r, train_m, test_r, test_m)

    # Evaluate the model
    predictions = model.predict(test_r)
    ndcg_score = call_ndcg(predictions, test_r)
    print(f"NDCG Score: {ndcg_score}")

if __name__ == "__main__":
    main()