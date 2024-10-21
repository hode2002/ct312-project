import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

"""### ĐỌC DỮ LIỆU"""

# 1. Đọc dữ liệu
#data_path = "Dry_Bean_Dataset2024.xlsx"
data_path = "https://github.com/ltdaovn/dataset/raw/master/Dry_Bean_Dataset2024.xlsx"
data = pd.DataFrame(pd.read_excel(data_path, sheet_name='Dry_Beans_Dataset'))

data = data.drop('ShapeFactor5', axis=1)
data = data.drop('Name', axis=1)

"""### KIỂM TRA VÀ XÓA DỮ LIỆU NULL"""
data = data.dropna()

"""### XỬ LÝ VÀ XÓA DỮ LIỆU KHÔNG NHẤT QUÁN"""
classList = ["SEKER", "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SIRA"]
data = data.loc[data['Class'].isin(classList)]

data = data.replace(',', '.', regex=True)
scaler = MinMaxScaler()
numeric_columns = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
    'Extent', 'Solidity', 'roundness', 'Compactness',
    'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
]

data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

def detect_outliers_iqr(data, column,factor):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

data_without_last_column = data.iloc[:, :-1]

smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)

x_resampled, y_resampled = smote.fit_resample(data_without_last_column, data['Class'])
data = pd.concat([x_resampled, y_resampled], axis=1)

# Bắt đầu với cột liên quan đến hình dạng và kích thước
data = detect_outliers_iqr(data,'Eccentricity', 0.9)
data = detect_outliers_iqr(data,'Solidity', 0.9)
data = detect_outliers_iqr(data,'roundness', 0.9)

# Tiếp theo là các cột kích thước
data = detect_outliers_iqr(data,'Area', 1.25)
data = detect_outliers_iqr(data,'MajorAxisLength', 1.25)
data = detect_outliers_iqr(data,'ConvexArea', 1.25)
data = detect_outliers_iqr(data,'Perimeter', 1)
data = detect_outliers_iqr(data,'MinorAxisLength', 1)

# Cuối cùng xử lý các Shape Factors
data = detect_outliers_iqr(data,'ShapeFactor4', 1.25)
data = detect_outliers_iqr(data,'ShapeFactor1', 0.9)
data = detect_outliers_iqr(data,'ShapeFactor2', 0.9)
data = detect_outliers_iqr(data,'ShapeFactor3', 0.9)

# Cuối cùng là Aspect Ratio
data = detect_outliers_iqr(data,'AspectRation', 1)
data = detect_outliers_iqr(data, 'Extent',1)
