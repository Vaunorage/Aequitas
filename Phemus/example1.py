import os, sys

import pandas as pd

from paths import HERE

os.chdir('..')
sys.path.append(os.getcwd())
from src.Phemus import *


def main():
    dataset_dir = "/home/vaunorage/PycharmProjects/ClearBias/discriminations.csv"
    df = pd.read_csv(dataset_dir)
    num_params = 8
    sensitive_param_idx = 0  # Starts at O
    sensitive_param_name = "Gender"
    col_to_be_predicted = "outcome_ou2"
    model_type = "DecisionTree"
    os.chdir(HERE.joinpath('Phemus/Examples/Synthetic').as_posix())
    sys.path.append(os.getcwd())

    dataset = Dataset(num_params=num_params, sensitive_param_idx=sensitive_param_idx, \
                      model_type=model_type, sensitive_param_name=sensitive_param_name, \
                      col_to_be_predicted=col_to_be_predicted, dataset_dir=dataset_dir)

    pkl_dir = 'Employee_DecisionTree_Original.pkl'
    improved_pkl_dir = 'Employee_DecisionTree_Original_Improved.pkl'
    retrain_csv_dir = 'Employee_Retraining_Dataset.csv'
    plot_dir = 'Employee_Fairness_Plot.png'

    perturbation_unit = 1

    num_trials = 100
    samples = 100
    global_iteration_limit = 1000  # needs to be at least 1000 to be effective
    local_iteration_limit = 100
    threshold = 0

    retrain_csv_dir = 'Employee_Retraining_Dataset.csv'
    run_aequitas(dataset=dataset, perturbation_unit=perturbation_unit,
                 pkl_dir=pkl_dir, improved_pkl_dir=improved_pkl_dir,
                 retrain_csv_dir=retrain_csv_dir, plot_dir=plot_dir, mode="Fully", threshold=threshold,
                 global_iteration_limit=global_iteration_limit, local_iteration_limit=local_iteration_limit,
                 num_trials=num_trials, samples=samples)

    # run_aequitas_once(dataset=dataset, perturbation_unit=perturbation_unit, pkl_dir=pkl_dir, improved_pkl_dir=improved_pkl_dir, retrain_csv_dir=retrain_csv_dir, plot_dir=plot_dir, mode="Fully")
    # retrain_sklearn(dataset, pkl_dir, retrain_csv_dir, improved_pkl_dir, plot_dir, num_trials, samples)
    # get_fairness_estimation(dataset, pkl_dir, threshold, num_trials, samples)


if __name__ == "__main__":
    main()
