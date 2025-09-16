# CHGNet_post-processing-script
Post-processing script for CHGNet

The CHGNet model comes from this study:

Deng, B., Zhong, P., Jun, K. et al. CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling. Nat Mach Intell 5, 1031–1041 (2023). 

https://doi.org/10.1038/s42256-023-00716-3

# Outline
- [特色](# Executable file: cnsub_run.py)

## Executable file: cnsub_run.py
### Input files:
**Input_CHGNet** : Required. If this file is not present, the program will generate a template and then terminate execution.

**POSCAR** : Required. Must conform to the standard VASP input format.

**Fine_Tune_Model.tar** : Optional. Provides parameter configurations during execution. If absent, the program defaults to the parameters of CHGNet v0.3.0.

### Output main files:
**CHGNet_results.log** : Output log of program execution.

**Output_details** : System details such as forces and tensions are output during program execution.

**Trajectory_VASP** : Trajectory files of atomic positions during relaxation are written in VASP format.


## Executable file: cnsub_grep.py
### Input files (from VASP)
Just follow user interaction instructions

### Output files
**dataset_*.json** : Training data captured by the program


## Executable file: cnsub_train.py
### Input files (from cnsub_grep.py)
**Input_CHGNet_Training** : Required. Training process settings. If this file is not present, the program will generate a template and then terminate execution.

**dataset_*.json** : Required. Training data set captured by cnsub_grep.py.

**Fine_Tune_Model.tar** : Optional. Provides parameter configurations during execution. If absent, the program defaults to the parameters of CHGNet v0.3.0.

### Output main files
**epochX_eX_fX_sX_mX.pth.tar** or 

**bestE_epochX_eX_fX_sX_mX.pth.tar** or

**bestF_epochX_eX_fX_sX_mX.pth.tar** : New model trained by the program.

**_CHGNet_X.png** : The loss function of the training set and the validation set output by the training process. With each epoch training output.

**__CHGNet_X.png** : Scatter plot of the test set, only output after training is completed.
