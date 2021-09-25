To train the deep learning recommendation models, we conduct a hyper-parameter
sweep using the open-source repository
(https://github.com/facebookresearch/dlrm). As the hyper-parameter sweep
(hundreds of networks, each taking 3 hours to train) can take a long time, we
provide output files from the hyper-parameter sweep in `model_outputs/`
directory.

Please run `python hyperparam.py` to parse the results from the hyper-parameter
sweep and print the Pareto-optimal models to the console.
