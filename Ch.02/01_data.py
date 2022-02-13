# %%
import os
import tarfile
import urllib.request as urllib
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# 1. 데이터 로드
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# fetch_housing_data()


# %%

# 2. 데이터 확인
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


housing = load_housing_data()
print(housing)
housing.head()

# %%
# 데이터 정보 확인
housing.info()

# %%
# 특정 범주 데이터 확인
housing['ocean_proximity'].value_counts()

# %%
# 데이터 통계값 확인
housing.describe()
# %%
# 히스토그램 출력

housing.hist(bins=50, figsize=(20,15))
plt.show()
# %%

# 3. 테스트 세트 만들기
# train 80%, test 20% 분류

# 방법 1
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data)) # len(data) = 20640, 랜덤으로 배열 생성
    test_set_size = int(len(data) * test_ratio) # 전체 데이터에서 몇 퍼?
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices] # 길이의 해당하는 각 행의 해당값만 리턴

train_set, test_set = split_train_test(housing, 0.2)
len(train_set)

# %%

# 방법 2.
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 * 32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')
len(train_set)

# %%

# 방법 3. sklearn

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
len(train_set)

# %%

housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
housing['income_cat'].hist()

# %%

# sklearn 으로 데이터 샘플링

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# %%

# 전체 비율에 맞게 샘플링도 그 비율을 맞춰야한다.
# 무작위로 샘플링을 하면 계층의 비율이 달라진다.
# 그러므로 전체 비율에 비슷하게 샘플링도 비슷하게 샘플링을 해야한다.
# p.90 참조

strat_test_set["income_cat"].value_counts() / len(strat_test_set)
# %%

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
# %%
