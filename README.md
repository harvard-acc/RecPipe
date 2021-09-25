# RecPipe: Co-designing Models and Hardware to Jointly Optimize Recommendation Quality and Performance
RecPipe provides an end-to-end system to study and jointly optimize recommendation models and hardware for at-scale inference. This repository provides the RecPipe infrastructure which is configuration across different models and hardware (general purpose and specialized accelerator) dimensions. The infrastructure enables judicious comparison of recommendation quality, inference latency, and system-throughput.

Central to RecPipe is the implementation of a multi-stage recommendation pipeline. At the model level, varying number of stages and network complexity enable tradeoffs between end-to-end quality and run-time performance based on application-level requirements. At the hardware-level, varying how each stage is mapped to general purpose CPUs and GPUs, as well as the design of backend specialized accelerators opens new opportunities to optimize multi-stage pipelines.

## Organization
The code is structured to enable researchers and developers to extend the infrastructure for future studies.

1. The top-level is found in ```RecPipeMain.py```. This co-ordinates the various RecPipe experiment sweeps based on model, hardware, and RecPipe configuration files.
2. Configuration files can be found in the ```configs``` directory. We separate them into json files for models, hardware, and the RecPipe infrastucture to ease extended each component separately.
3. The multi-stage recommendation infrastructure can be found in ```RecPipe.py```. This co-ordinates the multi-stage recommendation pipeline, at-scale load generator, recommendation models, and backend hardware.
4. The models are found in the ```models``` directory.
5. The modeled accelerator, RecPipeAccel, is found in the ```RecPipeModel.py```. To simulate the accelerator with the overall RecPipe infrasturcture, we provide ```RecPipeAccelSim.py```

You can build the necessary python packages, using conda or docker environments, based on build specifications found in ```docker``` directory.

## Getting started
To get you started quickly, we provide example scripts to run RecPipe on general purpose CPUs and GPUs, and specialized hardware (RecPipeAccel). You run the code base please use the following steps: 

1. Clone this repository
2. Build the docker image located in `docker` (`cd docker; ./build-docker.sh`)
3. Launch docker image in interactive mode
4. Download the Criteo Kaggle and/or MovieLens (1-million or 20-million) datasets
5. Update `DATA_DIR` environment variable in bash scripts in `scripts` directory based on local dataset location
6. Download pre-trained deep learning recommendation models or train using open-source repositories like [Facebook's DLRM](https://github.com/facebookresearch/dlrm)
7. Update the model locations in json configuration files for `dim_4_0.json`, `dim_16.json`, and `dim_32_0.json` files in `scripts/model_configs/` directory
8. From the repository’s root directory, run test script, `./scripts/recpipe_kaggle.sh`
9. You can reproduce results from the [RecPipe](https://arxiv.org/abs/2105.08820) paper by 
    1. Figure 2: Running python script in `plotting/figure2/`. This should only take a couple of minutes.
    2. Figure 3: Running `./scripts/artifact_eval/figure3.sh` from the root directory of the repository. To parse and print the final results please look at `plotting/figure3`. This should take less than 30 minutes.
    3. Figure 7 and 8: Running `./scripts/artifact_eval/figure3.sh` from the root directory of the repository. To parse and print the final results please look at `plotting/figure7` and `plotting/figure8`. This may take around 12 hours. 
    4. Figure 10a: Running `./scripts/artifact_eval/figure10a.sh` from the root directory of the repository. To parse and print the final results please look at `plotting/figure10a`. This may take up to 2 hours.
    5. Figure 10c: First download extacted embedding access trace from the Criteo Kaggle dataset `indices_lookup_small.npy`. Move `indices_lookup_small.npy` to `./RPAccel_characterization/embedding` and run `./scripts/artifact_eval/figure10c.sh` from the root directory of the repository. To parse and plot the final results please look at `plotting/figure10c`. This may take up to 6 hours.
    6. Figure 11: The logic synthesis reports of main RecPipeAccel components can be found at `RPAccel_characterization/Synthesis_Report/`. To see the area and power analysis please look at `plotting/figure11`.
    7. Figure 12: Running `./scripts/artifact_eval/figure12.sh` from the root directory of the repository. To parse and print the final results please look at `plotting/figure12`. This may take up to 6 hours.
    8. Figure 13: Running `./scripts/artifact_eval/figure13.sh` from the root directory of the repository. To parse and print the final results please look at `plotting/figure13`. This may take up to 6 hours.
    9. Figure 14: Having generated results for Figure 7, 8, and 12 already, you can generate results for Figure 14 (Criteo Kaggle) using the parsing scripts found in `plotting/figure14`. 

### General purpose hardware
To run on general purpose hardware (CPUs and GPUs) you may use the ```scripts/recpipe_kaggle.sh``` and ```scripts/recpipe_movielens.sh```.

### Specialized recommendation accelerators
To run on general purpose hardware (CPUs and GPUs) you may use the ```scripts/recpipe_kaggle_accel.sh```. Different backend accelerators can be evaluated by extending the accelerator model in ```RecPipeAccelModel.py```.

## Models and Datasets
RecPipe is built to evaluate the impact of different models on end-to-end quality and performance. We evaluate RecPipe on open-source Criteo Kaggle and MovieLens datasets. Due to copyright restrictions we are unable to provide copies of the dataset and these should be setup independent of RecPipe.

RecPipe can also be extended to consider different models and datasets by building on the implementations found in the ```models``` directory.

## Link to paper
To read the paper please visit this [link](https://arxiv.org/abs/2105.08820)

## Citation
If you use `RecPipe`, please cite our work:

```
@article{gupta2021recpipe,
  title={RecPipe: Co-designing Models and Hardware to Jointly Optimize Recommendation Quality and Performance},
  author={Gupta, Udit and Hsia, Samuel and Zhang, Jeff and Wilkening, Mark and Pombra, Javin and Lee, Hsien-Hsin S and Wei, Gu-Yeon and Wu, Carole-Jean and Brooks, David and others},
  journal={arXiv preprint arXiv:2105.08820},
  year={2021}
}
 ```

## Contact Us
For any further questions please contact Udit Gupta (<ugupta@g.harvard.edu>).
