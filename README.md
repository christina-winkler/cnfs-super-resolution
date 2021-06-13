# Conditional Normalizing Flows for Super-Resolution

This repository contains an implementation in Pytorch of conditional normalizing flows applied to super-resolution.
This master thesis project was conducted at the Amsterdam Machine Learning Lab 2019. 
Pre-print:
**[Likelihood Learning with Conditional Normalizing Flows](https://arxiv.org/abs/1912.00042)**.

Results of the conditional normalizig flow model compared to a factorized likelihood baseline:
![Image](https://github.com/christina-winkler/cnfs-super-resolution/blob/master/git_cnf_compare.png?raw=true)


### Requirements

The requirements for the conda environment in which we have tested this code are started in `environment_nfsr.yaml`.

The main dependencies are:
-   `python 3.8.3`
-   `pytorch 1.8.0` 

The conda environment can be installed via

	```
	conda env create -f environment_nfsr.yaml 
	```
	
And used by running

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

