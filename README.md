# NTUST-CHGNet_post-processing-script
Post-processing script for CHGNet

## Executable file: cnsub_run.py
### Input files:
**Input_CHGNet** : Required. If this file is not present, the program will generate a template and then terminate execution.

**POSCAR** : Required. Must conform to the standard VASP input format.

**Fine_Tune_Model.tar** : Optional. Provides parameter configurations during execution. If absent, the program defaults to the parameters of CHGNet v0.3.0.

### Output files:
**CHGNet_results.log** : Output log of program execution.

**Output_details** : System details such as forces and tensions are output during program execution.

**Trajectory_VASP** : Trajectory files of atomic positions during relaxation are written in VASP format.
