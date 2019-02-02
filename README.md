# ETL - Extract / Translate / Load
Provides a basic directory structure and template files for setting up a DataLoader using the ETL methodology.

## Basic Usage
```python
from etl.dataloader import DataLoader
dl = DataLoader()

train_gen = dl.retrieve_data(<ml_cfg>)
test_gen = dl.get_test_data()
valid_gen = dl.get_validation_data()
```

## Config File
The config file can be in either JSON or YAML format.  Fields are optional unless otherwise stated.

### Fields
 - __data_dir__: directory where data is located; path can be absolute or relative to directory of task.py
 - __batch_size__: number of records per batch
 - __epochs__: number of epochs to run through during training
 - __train_size__: decimal ratio of training data
 - __test_size__: decimal ratio of test data
 - __valid_size__: decimal ratio of validation data
