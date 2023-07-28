import os
import time
import csv
from pathlib import Path

import pandas as pd # pip install pandas
# pip install scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def data_import(csv_file):
    """
    csv_file (string) : 불러올 csv 파일 경로 (예시 ../data/titanic.csv), 
                        같은 위치에 있다면 파일 이름 (예시 - titanic.csv) 
    """
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        arr = [row for row in reader]
        columns = arr[0]
        values = arr[1:]
        df = pd.DataFrame(values, columns=columns)
    return df

def remove_cats(df):
    " df : 범주형 데이터를 제거할 dataFrame "
    cat_feats = df.dtypes[df.dtypes == "object"].index # col name
    df.drop(cat_feats, axis=1, inplace=True)
    return df

def data_split(df, target):
    """
    df : 데이터 분리할 dataFrame
    target (string) : 타겟 값의 column name
    """
    x= df.drop(target, axis=1)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    return x_train, x_val, x_test, y_train, y_val, y_test

def validation_data(df, file_name):
    """
    df : 내보낼 dataFrame
    file_name (string) : 저장할 경로
                         예시) ../data/binary_val(저장할 이름).csv
    """
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    filepath = Path(file_name)
    val_df.to_csv(filepath, index=False)

def scaling(x_train, x_val, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_val_scaled, x_test_scaled
