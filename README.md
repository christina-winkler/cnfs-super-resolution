# Normalizing Flows for Super-Resolution

This repository contains an implementation of my master thesis
project. 
Pre-print:
**[Likelihood Learning with Conditional Normalizing Flows](https://arxiv.org/abs/1912.00042)**.

### Requirements

The requirements for the conda environment in which we have tested this code are started in `environment_nfsr.yaml`.

The main dependencies are 
-   `python 3.8.3`
-   `pytorch 1.8.0` 
The conda environment can be installed via
	```
	conda env create -f environment_nfsr.yaml 
	```
And used by
	```
	source activate cnf-sr
	```

### Cite

Please cite our pre-print if you use this code in your own work:

```
@misc{winkler2019learning,
      title={Learning Likelihoods with Conditional Normalizing Flows}, 
      author={Christina Winkler and Daniel Worrall and Emiel Hoogeboom and Max Welling},
      year={2019},
      eprint={1912.00042},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

