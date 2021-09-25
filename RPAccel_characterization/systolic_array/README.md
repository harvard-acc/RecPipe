## Notes


1. This is based off the SCALE-Sim simulator developed by ARM (see `LICENSE`). Per SCALE-Sim's original description: "SCALE-Sim is a CNN accelerator simulator, that provides cycle-accurate timing, power/energy, memory bandwidth and trace results for a specified accelerator configuration and neural network architecture."

2. There are two wrapper files that we created for our characterization studies: `gen_models.py` and `run_models.py`

  1. `gen_models.py` generates model architecture configurations to be executed later by accelerators.

  2. `run_models.py` generates accelerator configurations and executes the model architectures specified in `gen_models.py`.