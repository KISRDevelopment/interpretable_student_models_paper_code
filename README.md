# Neural Bayesian Knowledge Tracing

This repository is associated with my paper of the same title. 
It implements Bayesian Knowledge Tracing (BKT) as a recurrent neural network in Pytorch and demonstrates three novel extensions to it:

- Accelerated BKT RNN which processes multiple trials at time in parallel.
- Multidimensional BKT+IRT which learns student prototypes from the training set and generalizes to new students in a principled way.
- Skill discovery model which learns the problem-KC assignments in an end-to-end way, while optionally incorporating problem features and an auxiliary blocked KC loss.

## File Description

### Models

- `model_bkt_irt.py`: implements BKT, BKT+Abilities (Unidimensional), BKT+Problems, and BKT+IRT (Unidimensional abilities). The models can use two alternative BKT RNN implementations interchangably (specified in the JSON configuration file):
    - `layer_bkt.py`: Standard BKT RNN layer 
    - `layer_fastbkt.py`: Accelerated BKT RNN layer which processes multiple trials at a time

To generalize to new students in a principled way, the model uses the sequential Bayesian updating layer in `layer_seq_bayesian.py`.

- `model_bkt_irt_multidim_abilities.py`: supports learning multidimensional student prototypes and sequential Bayesian updating over those phenotypes to generalize to new students. Otherwise, it is the same as `model_bkt_irt.py`.

- `model_sd_old.py`: The skill discovery model (the `_old` postfix is a relic that kinda stuck, please ignore it!). The model uses the following layers:
    - `layer_multihmmcell.py`: A multi-KC BKT cell that is differentiable with respect to the input KCs.
    - `layer_kc_discovery.py`: Stochastic NN layer that samples the problem-KC assignment matrix and supports using problem features as input to parameterize that matrix.
- `model_brute_force_bkt.py`: a brute-force BKT implementation (accelerated with the Numba library). 
- `ref_hmm.py`: a wrapper to call `hmm-scalable`.

An example of running a model from the command line:

`python model_bkt_irt.py cfgs/exp_model_comp/fbkt-irt.json gervetetal_assistments09 tmp/results.csv`
* `cfgs/exp_model_comp/fbkt-irt.json` is the configuration file of the model
* `gervetetal_assistments09` is the name of the dataset which should be located in `data/datasets` and the corresponding split in `data/splits`.
* `tmp/results.csv` where CV results will be deposited.

### Experiments

- `exp_perf.py`: runs performance and parameter recovery experiments on a single model:
`python exp_perf.py [cfg file path] [output dir]`

- `exp_sd.py`: runs the skill discovery experiments on a single model:
`python exp_sd.py [cfg file path] [output dir] [blocked|interleaving] [use_embeddings:0 or 1]`

- `exp_model_comp.py`: runs CV on the given model on all 8 datasets from Gervet et al. (2020), or a single dataset (optionally specified via a third command line argument):
`python exp_model_comp.py [cfg file path] [output dir] ([dataset_name])`

- `exp_interpret.py`: generates data to help interpret  multidimensional BKT+IRT model:
`python exp_interpret.py [cfg file path] [dataset name] [state dictionary file] [output file]`
The model state dictionary file has postfix `.state_dicts` and is saved by `model_bkt_irt_multidim_abilities.py`. 

### Synthetic data generators

- `generate_perf_data.py`: generates data for the performance experiment.
- `generate_skill_discovery_data.py`: generates data for the skill discovery experiment.

Generator scripts don't need any input parameters on the command line.

### Result generators

- `results_param_recovery.py [input directory] [output_file]`: generates a CSV file summarizing the parameter recovery results.
- `results_performance.py [input directory] [output_file]`: generates a CSV file summarizing the performance experiment results.
- `results_model_comp.py [input directory] [output file]`: generates a CSV file containing split-by-split model comparison experiment results.
- `results_sd.py [input directory] [output file]`: generates a CSV file containing split-by-split KC discovery experiment results.

### Data processing

- `process_gervetetal.ipynb`: processes and splits the datasets from [Gervet et al.](https://github.com/theophilee/learner-performance-prediction) Github repository.

- `split_dataset.py [dataset_path] [splits_path]`: splits a given dataset into 5 student-stratified folds, with each training fold being further split into 80\% optimization and 20\% validation.

## Requirements

- `pandas`
- `pytorch`
- `numpy`
- `scipy`
- `sklearn`
