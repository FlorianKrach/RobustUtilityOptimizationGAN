# Robust Utility Optimization GAN

[![DOI](https://zenodo.org/badge/764250490.svg)](https://zenodo.org/doi/10.5281/zenodo.10893701)

This repository is the official implementation of the paper [Robust Utility Optimization via a GAN Approach](https://arxiv.org/abs/2403.15243).

## Installation
Download the repo, cd into it. 
Create a new conda environment (if wanted): 
```shell
conda create --name RobustUtilityOptimization python=3.7
conda activate RobustUtilityOptimization
```
and install the needed libraries by:
```sh
pip install -r requirements.txt
```

## Running the code

### Dataset Generation
First generate the needed datasets.

```sh
python code/data_utils.py --data_gen_dict=data_dict1 --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict2 --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict3 --DATA_NB_JOBS=4

python code/data_utils.py --data_gen_dict=data_dict0_1 --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_1_test --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_1_test_large --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_2 --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_2_test --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_2_test_large --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_3 --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_3_test --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_3_test_large --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_4 --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_4_test --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_4_test_large --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_5 --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_5_test --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_5_test_large --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_6 --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_6_test --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_6_test_large --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_7 --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_7_test --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_7_test_large --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_8 --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_8_test --DATA_NB_JOBS=4
python code/data_utils.py --data_gen_dict=data_dict0_8_test_large --DATA_NB_JOBS=4
```

### Training of models
Use the parallel_train.py file in combination with the config.py file. 
First create the wanted configuration dicts in `config.py` and then use the command-line 
tools of `run.py` to run the training with these configurations.

Example:
```sh
python code/run.py --params=params_list1 --NB_JOBS=1 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict1
```

List of all flags:
- **params**: name of the params list (defined in config.py) to use for parallel run
- **NB_JOBS**: nb of parallel jobs to run with joblib
- **first_id**: First id of the given list / to start training of
- **get_overview**: name of the dict (defined in config.py) defining input for extras.get_training_overview
- **USE_GPU**: whether to use GPU for training
- **ANOMALY_DETECTION**: whether to run in torch debug mode
- **SEND**: whether to send results via telegram
- **NB_CPUS**: nb of CPUs used by each training
- **model_ids**: List of model ids to run
- **DEBUG**: whether to run parallel in debug mode
- **saved_models_path**: path where the models are saved
- **overwrite_params**: name of dict (defined in config.py) to use for overwriting params
- **N_DATASET_WORKERS**: number of processes that generate batches in parallel
- **evaluate_models**: name of the dict (defined in config.py) defining input for eval_pretrained_model.evaluate_models


run training (on server):
```shell
python code/run.py --params=params_list1 --NB_JOBS=40 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict1
```

### Evaluation:

run evaluation:
```shell
python code/run.py --evaluate_models=eval_model_dict12_last --NB_JOBS=40 --NB_CPUS=1 --get_overview=TO_dict12
```

### RUN the experiments
2dim 
```shell
python code/run.py --params=params_list_2dim --NB_JOBS=40 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_2dim --evaluate_models=eval_model_dict_2dim_best
python code/run.py --NB_JOBS=40 --NB_CPUS=1 --SEND=True --get_overview=TO_dict_2dim --evaluate_models=eval_model_dict_2dim_best_1
```

5dim
```shell
python code/run.py --params=params_list_5dim --NB_JOBS=40 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_5dim --evaluate_models=eval_model_dict_5dim_best
python code/run.py --NB_JOBS=40 --NB_CPUS=1 --SEND=True --get_overview=TO_dict_5dim --evaluate_models=eval_model_dict_5dim_best_1
```

2dim with Transaction Costs (with new noisy eval)
```shell
python code/run.py --params=params_list_2dim_TC --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_2dim_TC --evaluate_models=eval_model_dict_2dim_best_TC
python code/run.py --NB_JOBS=48 --NB_CPUS=1 --SEND=True --get_overview=TO_dict_2dim_TC --evaluate_models=eval_model_dict_2dim_best_1_TC
```

5dim with Transaction Costs (and noisy eval)
```shell
python code/run.py --params=params_list_5dim_TC --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_5dim_TC --evaluate_models=eval_model_dict_5dim_best_TC
python code/run.py --NB_JOBS=48 --NB_CPUS=1 --SEND=True --get_overview=TO_dict_5dim_TC --evaluate_models=eval_model_dict_5dim_best_1_TC
```

2dim with robust drift
```shell
python code/run.py --params=params_list_2dim_robd --NB_JOBS=40 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_2dim_robd --evaluate_models=eval_model_dict_2dim_best_robd
python code/run.py --NB_JOBS=40 --NB_CPUS=1 --SEND=True --get_overview=TO_dict_2dim_robd --evaluate_models=eval_model_dict_2dim_best_robd_1
```

2dim with robust drift -- comparison of models (RNN, FFNN, timegrid)
```shell
python code/run.py --params=params_list_2dim_robd_modelcomp --NB_JOBS=40 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_2dim_robd_modelcomp
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_2dim_best_robd_modelcomp
```

2dim with robust drift and power utility function
```shell
python code/run.py --params=params_list_2dim_robd_powU --NB_JOBS=40 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_2dim_robd_powU --evaluate_models=eval_model_dict_2dim_best_robd_powU
python code/run.py --NB_JOBS=40 --NB_CPUS=1 --SEND=True --get_overview=TO_dict_2dim_robd_powU --evaluate_models=eval_model_dict_2dim_best_robd_powU_1
```

2dim lambda grid
```shell
python code/run.py --params=params_list_2dim_lambdagrid --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_2dim_lambdagrid --evaluate_models=eval_model_dict_2dim_best_lambdagrid
```

2dim lambda grid -- comparison of models (RNN, FFNN, timegrid)
```shell
python code/run.py --params=params_list_2dim_lambdagrid_modelcomp --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_2dim_lambdagrid_modelcomp --evaluate_models=eval_model_dict_2dim_best_lambdagrid_modelcomp
```


2dim lambda-grid with path-wise penalty and const noise
```shell
python code/run.py --params=params_list_2dim_lambdagrid_pwp --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_2dim_lambdagrid_pwp --evaluate_models=eval_model_dict_2dim_best_lambdagrid_pwp
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_2dim_best_lambdagrid_pwp
```

2dim lambda-grid with path-wise penalty and non-const noise (for validation, i.e. model/epoch selection, and evaluation)
```shell
python code/run.py --params=params_list_2dim_lambdagrid_pwp1 --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_2dim_lambdagrid_pwp1 --evaluate_models=eval_model_dict_2dim_best_lambdagrid_pwp1_cum_sn
python code/run.py --NB_JOBS=48 --NB_CPUS=1 --SEND=True --get_overview=TO_dict_2dim_lambdagrid_pwp1 --evaluate_models=eval_model_dict_2dim_best_lambdagrid_pwp1_sn
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_2dim_best_lambdagrid_pwp1
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_2dim_best_lambdagrid_pwp1_baseline
```

2dim lambda-grid with path-wise penalty and non-const noise for eval -- comparison of models (RNN, FFNN, timegrid)
```shell
python code/run.py --params=params_list_2dim_lambdagrid_pwp1_modelcomp --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_2dim_lambdagrid_pwp1_modelcomp --evaluate_models=eval_model_dict_2dim_best_lambdagrid_pwp1_modelcomp_sn
```


2dim lambda-grid with path-wise penalty and GARCH eval
```shell
python code/run.py --params=params_list_2dim_lambdagrid_pwp_g --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_2dim_lambdagrid_pwp_g --evaluate_models=eval_model_dict_2dim_best_lambdagrid_pwp_g_ln
```

1dim lambda-grid with path-wise penalty and comparison to ref strategy for small transaction costs
```shell
python code/run.py --params=params_list_lambdagrid_pwp_smalltranscost1 --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_lambdagrid_pwp_smalltranscost1 --evaluate_models=eval_model_dict_lambdagrid_pwp_smalltranscost1
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost1_1
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost1_2
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost1_baseline
python code/run.py --NB_JOBS=1 --evaluate_models=eval_model_dict_lambdagrid_pwp_smalltranscost1_nonoise
```

```shell
python code/run.py --params=params_list_lambdagrid_pwp_smalltranscost2 --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_lambdagrid_pwp_smalltranscost2 --evaluate_models=eval_model_dict_lambdagrid_pwp_smalltranscost2
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost2_1
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost2_2
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost2_baseline
python code/run.py --NB_JOBS=1 --evaluate_models=eval_model_dict_lambdagrid_pwp_smalltranscost2_nonoise
```

```shell
python code/run.py --params=params_list_lambdagrid_pwp_smalltranscost3 --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_lambdagrid_pwp_smalltranscost3 --evaluate_models=eval_model_dict_lambdagrid_pwp_smalltranscost3
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost3_1
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost3_2
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost3_baseline
python code/run.py --NB_JOBS=1 --evaluate_models=eval_model_dict_lambdagrid_pwp_smalltranscost3_nonoise
```

```shell
python code/run.py --params=params_list_lambdagrid_pwp_smalltranscost4 --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_lambdagrid_pwp_smalltranscost4 --evaluate_models=eval_model_dict_lambdagrid_pwp_smalltranscost4_nonoise
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost4_1
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost4_baseline
```

```shell
python code/run.py --params=params_list_lambdagrid_pwp_smalltranscost5 --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_lambdagrid_pwp_smalltranscost5 --evaluate_models=eval_model_dict_lambdagrid_pwp_smalltranscost5
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost5_1
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost5_2
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost5_baseline
python code/run.py --NB_JOBS=1 --evaluate_models=eval_model_dict_lambdagrid_pwp_smalltranscost5_nonoise
```

```shell
python code/run.py --params=params_list_lambdagrid_pwp_smalltranscost6 --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_lambdagrid_pwp_smalltranscost6 --evaluate_models=eval_model_dict_lambdagrid_pwp_smalltranscost6_nonoise
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost6_1
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost6_baseline
```

```shell
python code/run.py --params=params_list_lambdagrid_pwp_smalltranscost7 --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_lambdagrid_pwp_smalltranscost7 --evaluate_models=eval_model_dict_lambdagrid_pwp_smalltranscost7_nonoise
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost7_1
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost7_baseline
```

```shell
python code/run.py --params=params_list_lambdagrid_pwp_smalltranscost8 --NB_JOBS=48 --NB_CPUS=1 --SEND=True --first_id=1 --get_overview=TO_dict_lambdagrid_pwp_smalltranscost8 --evaluate_models=eval_model_dict_lambdagrid_pwp_smalltranscost8_nonoise
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost8_1
python code/run.py --NB_JOBS=1 --evaluate_models=plot_model_dict_lambdagrid_pwp_smalltranscost8_baseline
```


--------------------------------------------------------------------------------
## Usage, License & Citation

This code can be used in accordance with the [LICENSE](LICENSE).

If you find this code useful or include parts of it in your own work, 
please cite our paper:  

- [Robust Utility Optimization via a GAN Approach](https://arxiv.org/abs/2403.15243)
    ```
    @article{RobustUO,
      url = {https://arxiv.org/abs/2403.15243},
      author = {Krach, Florian and Teichmann, Josef and Wutte, Hanna},
      title = {Robust Utility Optimization via a {GAN} Approach},
      publisher = {arXiv},
      year = {2024},
    }
    ```
