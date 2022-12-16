import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython import display
import pandas as pd
import numpy as np


def preprocessing(
    path_to_data = 'data/test.csv'
    ):

    ### acqusition and cleaning data ###

    # get data
    raw_data = pd.read_csv(path_to_data, encoding='utf-8', sep=',')

    print(raw_data['topic'].value_counts())
    raw_data.groupby("topic").topic.hist(bins=6)
    plt.title(f'Class Distribution in dataset')
    plt.xticks(rotation=45)
    plt.ylabel('counts')

    # cleaning data
    headers_list = list(raw_data)
    def replace(x, text_part):
        return x.replace(text_part, '')

    raw_data[headers_list[1]] = raw_data[headers_list[1]].apply(replace, text_part = '<?xml version="1.0" encoding="utf-8"?>\n<conversion><person>')
    raw_data[headers_list[1]] = raw_data[headers_list[1]].apply(replace, text_part = '</person></conversion>')

    return raw_data




