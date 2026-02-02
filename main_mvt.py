from config import ConfigCGC, set_seed
from mvt.custom_dataset_mvt import MultiTaskDataset
import os
import torch
from mvt.run_mvt import RunMVT
import panel as pn
from plot_results.plot_results import (
    plot_accuracy_across_seeds,
    plot_accuracy_loss_across_seeds,
    plot_winrate_boxplots,
    class_name_map,
    print_mean_std_metrics,
)
from data.download_dataset import download_and_unzip


################################################################################################################################################
def main():

    # get the seed
    base_seed = 42
    number_experiments = 10
    seeds = [base_seed + i for i in range(number_experiments)]

    # dict saved metrics
    dict_metrics_seed = {}
    dict_win_rate_seed = {}

    for seed in seeds:
        dict_metrics_seed[seed] = {}
        dict_win_rate_seed[seed] = {}

    # loop over the seeds
    for seed in seeds:

        # load the config
        cf = ConfigCGC(seed=seed)

        # download dataset
        download_and_unzip(cf=cf)

        # set the seed
        set_seed(cf.seed)

        # run the experiment
        run_mvt = RunMVT(cf)

        # train and get the stats
        (
            dict_win_rate_train,
            dict_win_rate_test,
            dict_metrics_train,
            dict_metrics_test,
        ) = run_mvt.get_win_rate_and_metrics_dict()

        # save dict
        dict_win_rate_seed[seed]["train"] = dict_win_rate_train
        dict_win_rate_seed[seed]["test"] = dict_win_rate_test
        dict_metrics_seed[seed]["train"] = dict_metrics_train
        dict_metrics_seed[seed]["test"] = dict_metrics_test

    # print the stats
    print_mean_std_metrics(
        dict_metrics_seed=dict_metrics_seed,
        dict_win_rate_seed=dict_win_rate_seed,
        class_name_map=class_name_map,
    )

    # fig accuracy
    fig_accuracy = plot_accuracy_across_seeds(
        dict_metrics_seed=dict_metrics_seed,
        cf=cf,
    )
    fig_accuracy.show()

    # fig loss accuracy
    fig_loss_accuracy = plot_accuracy_loss_across_seeds(
        dict_metrics_seed=dict_metrics_seed,
        cf=cf,
    )
    fig_loss_accuracy.show()

    # fig box plot
    fig_box = plot_winrate_boxplots(
        dict_win_rate_seed=dict_win_rate_seed,
        class_name_map=class_name_map,
        cf=cf,
    )
    fig_box.show()

    # fig tko
    fig_tko = run_mvt.run_tko_single_image()
    fig_tko.show()


if __name__ == "__main__":
    main()
