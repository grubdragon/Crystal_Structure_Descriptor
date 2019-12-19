# Project 2 - Machine Learning

## Crystal Structure Descriptor for Machine Learning

Stores code for our experiments with a simple crystal structure descriptor for machine learning on binary materials

## Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install any required packages for the project. You will need [LightGBM](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html), [TensorFlow](https://www.tensorflow.org/install), [sklearn](https://scikit-learn.org/stable/install.html), [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html), and [jupyter](https://jupyter.org/install). Note that future iterations of the project will require PyTorch too. Please look up their respectives websites to install them.

## Running the project


## Project Structure
Model files:
```
-energy_models
--4Layer_TF_binary.ipynb
--LGBM_Boosted_Trees.ipynb
--Linear Regression.ipynb
-conduction_models
--4Layer_net_TF.ipynb
--LGBM_Boosted_Trees.ipynb
--LogisticRegression.ipynb
```

Data files:
```
-data
--descriptor
---DescriptorData.csv
--pkls
---binary_cell_df.pkl
---binary_species_list.pkl
```

Extras(Plots and others):
```
README.md
-energy_models
--Accuracy.pdf
-conduction_models
--Plotting.ipynb
--ROC_binary.pdf
```

Utility files:
```
-data_processing
--descriptor.py
--descriptorExample.ipynb
--descriptor_run.ipynb
-helpers
--data_utils.py
--picklers.py
```
