Hydrate Saturation Prediction Using PSO-XGBoost
Machine learning prediction of gas hydrate saturation using PSO-XGBoost and rock physics constraints.

1. Project Description
This project provides MATLAB code for gas hydrate saturation prediction using the Particle Swarm Optimization (PSO) algorithm combined with XGBoost regression.
The model uses well-log data and rock-physics-derived information as input constraints to predict gas hydrate saturation. The project includes the MATLAB source code, public W17 well-log data, and fixed data partition indices required to reproduce the model training, validation, and testing procedures.

2. Repository Structure
├── code/
│   └── Main MATLAB program
│
├── data/
│   ├── W17_public.xlsx
│   ├── train.mat
│   └── README
│
├── model/
│   └── Reserved for trained model files
│
└── third_party/
    └── xgboost/
        ├── xgboost.h
        ├── xgboost_train.m
        └── xgboost_test.m

3. Data
"W17_public.xlsx" contains the public W17 well-log data used in this project.
For data privacy, the original depth information has been anonymized. The first column is retained as a sequential sample index for sample identification and visualization.

Columns 2–7: input features
Column 8: hydrate saturation

"train.mat" contains the fixed data partition indices:
"trainIdx": training set indices
"valIdx": validation set indices
"testIdx": testing set indices
These fixed indices are provided to ensure reproducibility.

4. Software Requirements
MATLAB
XGBoost MATLAB interface
A compatible "xgboost.dll"
The code was developed and tested using MATLAB R2024a.

5. XGBoost DLL
The "xgboost.dll" file is not included in this repository because its file size exceeds GitHub's 100 MB single-file limit.
Please download the required DLL separately and place it in:
third_party/xgboost/xgboost.dll

DLL download page:
https://github.com/Time9Y/Matlab-Machine/releases/tag/v1.0.0
After downloading the DLL, the "third_party/xgboost" folder should contain:

third_party/
└── xgboost/
    ├── xgboost.dll
    ├── xgboost.h
    ├── xgboost_train.m
    └── xgboost_test.m

6. Running the Code
(1). Download or clone this repository.
(2). Download "xgboost.dll" from the link provided above.
(3). Place "xgboost.dll" in "third_party/xgboost/".
(4). Open MATLAB.
(5). Open the main MATLAB program located in the "code" folder.
(6). Run the program.
The program automatically identifies the project directory and loads the data and XGBoost library using relative paths.

7. Reproducibility
The repository provides the public dataset and fixed training, validation, and testing indices used in the experiments.
The PSO algorithm is used to optimize the main XGBoost hyperparameters, including:
Learning rate (eta)
Maximum tree depth (max_depth)
Number of trees
L2 regularization (lambda)
L1 regularization (alpha)
Row subsampling (subsample)
Feature subsampling (colsample_bytree)
The final model is evaluated using RMSE, R², MAE, and MBE.

8. Notes
The W17 public dataset is provided for reproducibility and demonstration purposes. Original confidential or non-public well data are not included in this repository.
