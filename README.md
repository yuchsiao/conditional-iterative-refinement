# Conditional Iterative Refinement for Question Answering

This work is summarized in the report [here](/doc/report.pdf).
The code for processing [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) is taken from this
[repo](https://github.com/chrischute/squad).

## DISCLAIMER

This is a work in progress and it is nowhere in a status for shipping but merely for sharing for references.
Many of legacy parts are included without being removed for compatibility for old experiments.
Old experiment setups are found in the [exp](exp) folder.

## Requirements

* Create a conda environment using `environment.yml`

    ```sh
    conda env create -f environment.yml
    ```

* Run `setup.py` to download the dataset, GloVe embedding file, and SpaCy model files.

    ```sh
    python setup.py
    ```

## Model Training and Evaluation

```sh
python hb_train.py --output_dir <FOLDER>
```

Please also refer to the `exp` folder for hyperpamater choices.
The predictions for dev and test sets are generated at the end of each epoch.

## Experiment Results

Please refer to the [spreadsheet](exp/expbook.xlsx).



