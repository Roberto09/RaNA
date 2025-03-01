import os
from datasets import load_dataset
import pandas as pd
import os
from datasets import load_dataset, Dataset
import pandas as pd

def to_dataset(iterable_dataset):
    data_list = [item for item in iterable_dataset]
    dataset = Dataset.from_dict({key: [dic[key] for dic in data_list] for key in data_list[0]})
    return dataset

def get_baseline_dataset_v2(dataset_name, filename="./baseline_dataset.pkl", train_size=50_000, valid_size=2_000):
    if os.path.isfile(filename):
        print("reading pickle")
        return pd.read_pickle(filename)
    dataset = load_dataset(dataset_name)
    dataset = dataset["train"].train_test_split(train_size=train_size, test_size=valid_size, shuffle=True, seed=123)
    pd.to_pickle(dataset, filename)
    return dataset
