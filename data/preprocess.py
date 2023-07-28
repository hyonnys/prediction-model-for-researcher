import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# class Preprocess:
#     def __init__(self, args):
#         self.args = args
#         self.x_train = None
#         self.x_val = None
#         self.x_test = None
#         self.y_train = None
#         self.y_val = None
#         self.y_test = None

    # def get_train_test_data(self):
    #     return self.train_data, self.val_data, self.test_data

def data_import():
    return

def remove_cats(df):
    cat_feats = df.dtypes[df.dtypes == "object"].index # col name
    df.drop(cat_feats, axis=1, inplace=True)
    return df

# def split_data(self, df, target):
#     features = df.columns.difference([target])
#     y = df[target]
#     x = df[[features]]
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
#     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
#     scaler = StandardScaler()
#     x_train_scaled = scaler.fit_transform(x_train)
#     x_val_scaled = scaler.transform(x_val)
#     x_test_scaled = scaler.transform(x_test)
#     # x_train, x_val, x_test, y_train, y_val, y_test = self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test
#     return x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test

def data_split(df, target):
    x= df.drop(target, axis=1)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    return x_train, x_val, x_test, y_train, y_val, y_test

def scaling(x_train, x_val, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_val_scaled, x_test_scaled



