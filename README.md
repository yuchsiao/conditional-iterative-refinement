# Conditional Iterative Refinement for Question Answering

For the SQuAD task, the common start and end probability prediction strategy
for predicting the answer span may not be the thorough use of overall information.
We propose to iteratively exploit the hidden states to improve the prediction performance,
hence calling this method: Conditional Iterative Refinement. 

A corresponding training scheme is developed to improve the training convergence.
In our experiment, we observed a 2.0% improvement for the average F-1 score and a 2.5% improvement
for the exact match metric, achieving 80.65 and 77.76, respectively.

This work is summarized in the report [here](/doc/report.pdf).
The code for processing [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) is taken from this
[repo](https://github.com/chrischute/squad).

## DISCLAIMER

This is a work in progress and it is nowhere in a status for shipping but merely for sharing for references.
Many of legacy parts are included without being removed for compatibility for old experiments.
Old experiment setups are found in the [exp](exp) folder.

## Requirements

Create a conda environment using `environment.yml` and activate it.

```sh
conda env create -f environment.yml
conda activate cir
```

## Model Training and Evaluation

Run the data preprocessing first:

```sh
python setup.py
```

Train the model:

```sh
python hb_train.py --output_dir <OUTPUT_DIR>
```

Please also refer to the `exp` folder for hyperpamater choices.
The predictions for dev and test sets are generated at the end of each epoch.

## Experiment Results

Please refer to the [spreadsheet](exp/expbook.xlsx).



