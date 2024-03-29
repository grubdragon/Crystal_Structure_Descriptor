# Project 2 - Machine Learning

## Crystal Structure Descriptor for Machine Learning

Stores code for our experiments with a simple crystal structure descriptor for machine learning on binary materials

## Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install any required packages for the project. You will need [LightGBM](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html), [TensorFlow](https://www.tensorflow.org/install), [sklearn](https://scikit-learn.org/stable/install.html), [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html), and [jupyter](https://jupyter.org/install). Note that future iterations of the project will require PyTorch too. Please look up their respectives websites to install them.

## Running the project

There are three different machine learning models implemented for each of the two tasks. Please refer to the Project Structure for the following section.

Folder 'conduction_models' includes 4-layer neural network using TensorFlow in '4Layer_TF_binary.ipynb', LightGBM method in 'LGBM_Boosted_Trees.ipynb', and logistic regression in 'LogisticRegression.ipynb'. These models are used for binary classification of materials as conductors/insulators.

The predictions on the formation energies of the materials are obtained using 4-layer neural network in '4Layer_net_TF.ipynb', LightGBM method in 'LGBM_Boosted_Trees.ipynb', and linear regression in 'Linear Regression.ipynb' from the 'energy_models' folder.

All the models run using the data processed by our crystal-structure descriptor from file 'Descriptor.csv' at folder 'data/descriptor'. This file can be re-generated by running 'descriptor_run.ipynb' file from 'data_processing' folder. The initial data is provided in files 'binary_cell_df.pkl' and 'binary_species_list.pkl' in the folder 'data/pkls'.

The original data was fetched from the Materials Project database (https://materialsproject.org/). We fetch the following properties for all binary compounds ("nelements":2): 'energy_per_atom', 'band_gap', 'initial_structure'.


## Project Structure
Model files:
```
├── energy_models
│   ├── 4Layer_net_TF.ipynb
│   ├── LGBM_Boosted_Trees.ipynb
│   └── Linear Regression.ipynb
└── conduction_models
    ├── 4Layer_TF_binary.ipynb
    ├── LGBM_Boosted_Trees.ipynb
    └── LogisticRegression.ipynb
```

Data files:
```
└── data
    ├── descriptor
    |   └── DescriptorData.csv
    └── pkls
        ├── binary_cell_df.pkl
        └── binary_species_list.pkl
```

Extras(Plots and others):
```
├── README.md
├── energy_models
│   └── Accuracy.pdf
└── conduction_models
    ├── Plotting.ipynb
    └── ROC_binary.pdf
```

Utility files:
```
├── data_processing
|   ├── descriptor.py
|   ├── descriptorExample.ipynb
|   └── descriptor_run.ipynb
└── helpers
    ├── data_utils.py
    └── picklers.py
```
