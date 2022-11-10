# Neural Bayesian Knowledge Tracing

This repository is associated with my paper of the same title. 
It implements Bayesian Knowledge Tracing (BKT) as a recurrent neural network in Pytorch and demonstrates how this makes it easy to add various extensions to the model.

## File Description

### Models

- `torch_bkt.py`: standard BKT
- `torch_irt.py`: BKT-IRT 
- `torch_sd.py`: BKT-SD
- `torch_bkt_abilities.py`: BKT with student abilities inference
- `torch_bkt_problems.py`: BKT with problem effects
- `ref_hmm.py`: wrapper that calls BKT implementation in [hmm-scalable](https://github.com/myudelson/hmm-scalable).
- `exp_rep_learning.py`: implements BKT with representation learning and runs the corresponding experiments.
- `model_brute_force_bkt.py`: an accelerated brute-force BKT implementation (via the Numba library). Should be called from command line like this: `python model_brute_force_bkt.py input_path splits_path output_path`.

An example of running a model from the command line:

`python torch_bkt.py cfgs/bkt.json gervetetal_assistments09 tmp/results.csv`
* `cfgs/bkt.json` is the configuration file of the model
* `gervetetal_assistments09` is the name of the dataset which should be located in `data/datasets` and the corresponding split in `data/splits`.
* `tmp/results.csv` where CV results will be deposited.

### Experiments

- `exp_perf.py`: performance and parameter recovery experiments
- `exp_rep_learning.py`: implements BKT with representation learning and runs the corresponding experiments.
- `exp_skill_discovery.py`: skill discovery experiments
- `exp_model_comp.py`: model comparison experiments

Experiment scripts don't need any input parameters on the command line.

### Synthetic data generators

- `generate_dt.py`: generates representation learning problem instances (the high-dimensional inputs and associated difficulties).
- `generate_perf_data.py`: generates data for the performance experiment.
- `generate_replearning_data.py`: generates data for the representation learning experiment.
- `generate_skill_discovery_data.py`: generates data for the skill discovery experiment.

Generator scripts don't need any input parameters on the command line.

### Result generators

- `results_param_recovery.py [input directory] [output_file]`: generates a CSV file summarizing the parameter recovery results.
- `results_performance.py [input directory] [output_file]`: generates a CSV file summarizing the performance experiment results.
- `results_replearning.py [input directory] [output_file]`: generates a CSV file summarizing the representation learning results.
- `results_model_comp.py [input directory] [output file]`: generates a CSV file summarizing the model comparison experiment.

### Data processing

- `process_gervetetal.ipynb`: processes and splits the datasets from [Gervet et al.](https://github.com/theophilee/learner-performance-prediction) Github repository.

- `split_dataset.py [dataset_path] [splits_path]`: splits a given dataset into 5 student-stratified folds, with each training fold being further split into 80\% optimization and 20\% validation.

## Requirements

- `pandas`
- `pytorch`
- `numpy`
- `scipy`
- `sklearn`
