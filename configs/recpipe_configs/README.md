# RecPipe configuration files

This directory holds the RecPipe configuration files which specify the main experimental parameters to sweep in RecPipe. The different parameters can be set to sweep distinct design spaces as well as scale the size of experiments based on avaialble resources. Below we provide some guidelines on how to set the some of the parameters: 

- `num_queries`: The nmber of queries specifies the number of queries ranked during the at-scale simulation. For CPU and GPU configurations the minimum should be 1000 for repeatable experiments. For accelerator configurations the minimum is 10000.
- `nepochs`: The number of epochs or independent trials conducted for each configuration. Minimum should be 1.
- `max_cpu_engines`: The number of CPU processes to parallelize query processing over. Should be set to the number of physical cores available in your system (excluding hyper-threading)
- `arrival_rates`: The different query arrival rates (based on Poisson arrival) to sweep in load generator
- `model_configs`: The location of model configuration files with respect to the root level of the RecPipe repository
- `num_threads_per_engine`: Number of thread per CPU process. Should be set to 1. 
- `stage_batch_sizes`: The batch size, or number of items ranked per stage. 
- `num_stages`: Number of stages to run for multi-stage recommendation pipeline.
