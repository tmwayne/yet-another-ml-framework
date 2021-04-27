Yet Another Machine Learning Framework
=================================

This code is a set of classes to create a robust ML pipeline.
It provides logging and exception handling to automate
model training and scoring.

- `extract_model_data` loads the modeling data from a database
into memory for use

- `build_model_pipeline` creates training and validation pipeline.
It supports out-of-time sampling, hyperparameter tuning, feature selection,
and other features that aren't available out-of-the-box with `sklearn`

- `helper_functions` has, in addition to some utility functions,
a set of functions to create model inventories which can be
queried at the command line and make loading the best model
easier from inside the scoring script

- `score_customers` scores customers and uses templates to save
results to a database.

