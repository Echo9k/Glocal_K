from tensorflow import keras
import numpy as np
from data_loader import load_data_100k, load_data_1m, load_data_monti
from model import create_model, train_model
from evaluate import call_ndcg

def main():
    data_path = ''  # Specify your data path here
    dataset = 'ML-100K'  # Choose among 'ML-100K', 'ML-1M', 'Douban'

    # Load the dataset
    if dataset == 'ML-100K':
        path = data_path + 'data/MovieLens_100K/'
        n_m, n_u, train_r, train_m, test_r, test_m = load_data_100k(path=path, delimiter='\t')

    elif dataset == 'ML-1M':
        path = data_path + '/MovieLens_1M/'
        n_m, n_u, train_r, train_m, test_r, test_m = load_data_1m(path=path, delimiter='::', frac=0.1, seed=1234)

    elif dataset == 'Douban':
        path = data_path + '/Douban_monti/'
        n_m, n_u, train_r, train_m, test_r, test_m = load_data_monti(path=path)

    else:
        raise ValueError("Invalid dataset selection")

    # Build and compile the model
    n_hid = 500
    n_dim = 5
    gk_size = 3
    dot_scale = 1 # Add the dot_scale parameter
    model = create_model(n_m, n_u, n_hid, n_dim, gk_size, dot_scale) # Pass dot_scale to the function

    # Train the model
    train_model(model, train_r, train_m, epochs=30)

    # Evaluate the model
    predictions = model.predict(test_r)
    ndcg_score = call_ndcg(predictions, test_r)
    print(f"NDCG Score: {ndcg_score}")

if __name__ == "__main__":
    main()