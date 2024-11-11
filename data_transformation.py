import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import re

"""### ĐỌC DỮ LIỆU"""

# 1. Đọc dữ liệu
data_path = "https://github.com/ltdaovn/dataset/raw/master/Dry_Bean_Dataset2024.xlsx"
data = pd.DataFrame(pd.read_excel(data_path, sheet_name='Dry_Beans_Dataset'))

data = data.drop('ShapeFactor5', axis=1)
data = data.drop('Name', axis=1)

### KIỂM TRA VÀ XỬ LÝ DỮ LIỆU BỊ NULL
data_x = data.drop('Class', axis=1)
data_y = data['Class']

def is_invalid(value):
    if isinstance(value, str) and re.search(r'[^\d\.]', value):
        return True
    return False

for column in data_x:
    data_x[column] = data_x[column].where(~data_x[column].apply(is_invalid), np.nan)
data_x = data_x.apply(pd.to_numeric, errors='coerce')
data_x.fillna(data_x.mean(), inplace=True)
data = pd.concat([data_x, data_y], axis=1)

"""### XỬ LÝ VÀ XÓA DỮ LIỆU KHÔNG NHẤT QUÁN"""
classList = ["SEKER", "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SIRA"]
data = data.loc[data['Class'].isin(classList)]

### XỬ LÝ DỮ LIỆU BỊ NHIỄU (NGOẠI LAI)
def remove_outliers(df):
    for col in df.columns:
        if col != 'Class':
            q25 = np.percentile(df[col] , 25)
            q75 = np.percentile(df[col] , 75)
            iqr = q75 - q25
            cut_off = iqr * 1.5
            lo = q25 - cut_off
            up = q75 + cut_off
            df[col] = df[col].clip(upper = up)
            df[col] = df[col].clip(lower=lo)

remove_outliers(data)

###CÂN BẰNG DỮ LIỆU
X = data.drop('Class', axis=1)
y = data['Class']

smote = SMOTE(random_state=42)
x_smote, y_smote = smote.fit_resample(X, y)

###XÓA BỎ CÁC ĐẶC TRƯNG BẤT THƯỜNG
x_smote.drop(['ConvexArea', 'EquivDiameter','ShapeFactor3'], axis=1, inplace=True)
