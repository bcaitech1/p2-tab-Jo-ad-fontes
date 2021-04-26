# Suppress warnings
import warnings

warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Custom library
from utils import seed_everything, print_score

# group by aggregation 함수 선언
AGGREGATION_FUNCTIONS = ["mean", "max", "min", "sum", "std", "skew"]

# 구매액 임계값
TOTAL_THRES = 300

# 랜덤 시드 & 시드 고정
SEED = 42
seed_everything(SEED)

data_dir = "../input/train.csv"  # os.environ['SM_CHANNEL_TRAIN']
model_dir = "../model"  # os.environ['SM_MODEL_DIR']


"""
    입력인자로 받는 year_month에 대해 고객 ID별로 총 구매액이
    구매액 임계값을 넘는지 여부의 binary label을 생성하는 함수
"""


def generate_label(df, year_month, total_thres=TOTAL_THRES, print_log=False):
    df = df.copy()

    # year_month에 해당하는 label 데이터 생성
    df["year_month"] = df["order_date"].dt.strftime("%Y-%m")
    df.reset_index(drop=True, inplace=True)

    # year_month 이전 월의 고객 ID 추출
    cust = df[df["year_month"] < year_month]["customer_id"].unique()
    # year_month에 해당하는 데이터 선택
    df = df[df["year_month"] == year_month]

    # label 데이터프레임 생성
    label = pd.DataFrame({"customer_id": cust})
    label["year_month"] = year_month

    # year_month에 해당하는 고객 ID의 구매액의 합 계산
    grped = df.groupby(["customer_id", "year_month"], as_index=False)[["total"]].sum()

    # label 데이터프레임과 merge하고 구매액 임계값을 넘었는지 여부로 label 생성
    label = label.merge(grped, on=["customer_id", "year_month"], how="left")
    label["total"].fillna(0.0, inplace=True)
    label["label"] = (label["total"] > total_thres).astype(int)

    # 고객 ID로 정렬
    label = label.sort_values("customer_id").reset_index(drop=True)
    if print_log:
        print(f"{year_month} - final label shape: {label.shape}")

    return label


def feature_preprocessing(train, test, features, do_imputing=True):
    train_set = train.copy()
    test_set = test.copy()

    # 범주형 피처 이름을 저장할 변수
    category_cols = []

    # 레이블 인코딩
    for f in features:
        if train_set[f].dtype.name == "object":  # 데이터 타입이 object(str)이면 레이블 인코딩
            category_cols.append(f)
            le = LabelEncoder()
            # train + test 데이터를 합쳐서 레이블 인코딩 함수에 fit
            le.fit(list(train_set[f].values) + list(test_set[f].values))

            # train 데이터 레이블 인코딩 변환 수행
            train_set[f] = le.transform(list(train_set[f].values))

            # test 데이터 레이블 인코딩 변환 수행
            test_set[f] = le.transform(list(test_set[f].values))

    print("categorical feature:", category_cols)

    if do_imputing:
        # 중위값으로 결측치 채우기
        imputer = SimpleImputer(strategy="median")

        train_set[features] = imputer.fit_transform(train_set[features])
        test_set[features] = imputer.transform(test_set[features])

    return train_set, test_set


def new_cols(df_agg):
    # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
    cols = []
    for col in df_agg.columns.levels[0]:
        for stat in df_agg.columns.levels[1]:
            cols.append(f"{col}-{stat}")
    return cols


def new_cols_dict(agg_dict):
    # key-value로 컬럼명 변경
    cols = []
    for col in agg_dict.keys():
        for stat in agg_dict[col]:
            cols.append(f"{col}-{stat}")
    return cols


def feature_agg(df):
    df_agg = df.groupby(["customer_id"]).agg(AGGREGATION_FUNCTIONS)

    return df_agg


def feature_cumsum(df):
    df.product_id = df.product_id.str.slice(start=0, stop=2)
    df["cumsum_total_by_cust_id"] = df.groupby(["customer_id"])["total"].cumsum()

    df["cumsum_quantity_by_cust_id"] = df.groupby(["customer_id"])["quantity"].cumsum()

    df["cumsum_price_by_cust_id"] = df.groupby(["customer_id"])["price"].cumsum()

    df["cumsum_total_by_prod_id"] = df.groupby(["product_id"])["total"].cumsum()

    df["cumsum_quantity_by_prod_id"] = df.groupby(["product_id"])["quantity"].cumsum()

    df["cumsum_price_by_prod_id"] = df.groupby(["product_id"])["price"].cumsum()

    df["cumsum_total_by_order_id"] = df.groupby(["order_id"])["total"].cumsum()

    df["cumsum_quantity_by_order_id"] = df.groupby(["order_id"])["quantity"].cumsum()

    df["cumsum_price_by_order_id"] = df.groupby(["order_id"])["price"].cumsum()

    cols = [col for col in df.columns if "cumsum" in col]

    df_agg = df.groupby(["customer_id"])[cols].agg(AGGREGATION_FUNCTIONS)

    return df_agg


def df_diff(df, prev_ym, year_month):
    df["order_ts"] = df["order_date"].astype(np.int64) // 1e9
    df["order_ts_diff"] = df.groupby(["customer_id"])["order_ts"].diff()
    df["quantity_diff"] = df.groupby(["customer_id"])["quantity"].diff()
    df["price_diff"] = df.groupby(["customer_id"])["price"].diff()
    df["total_diff"] = df.groupby(["customer_id"])["total"].diff()

    # df['order_ts_diff'].fillna(df.order_ts, inplace=True)
    # df["quantity_diff"].fillna(df.quantity, inplace=True)
    # df["price_diff"].fillna(df.price, inplace=True)
    # df["total_diff"].fillna(df.total, inplace=True)

    train = df[df["order_date"] < prev_ym]
    test = df[df["order_date"] < year_month]

    return train, test


def feature_time_diff(df):
    agg_dict = {
        "order_ts": ["first", "last"],
        "order_ts_diff": ["mean", "max", "min", "sum", "count", "std", "skew"],
        "quantity_diff": AGGREGATION_FUNCTIONS,
        "price_diff": AGGREGATION_FUNCTIONS,
        "total_diff": AGGREGATION_FUNCTIONS,
    }

    df_agg = df.groupby(["customer_id"]).agg(agg_dict)
    return df_agg, agg_dict


def order_diff(df):
    order_df = df.groupby(["order_date", "customer_id"]).sum()
    order_df = order_df.groupby(["customer_id"]).diff()
    order_df["order_quantity_diff"] = order_df["quantity"]
    order_df["order_price_diff"] = order_df["price"]
    order_df["order_total_diff"] = order_df["total"]
    agg_dict = {
        "order_quantity_diff": AGGREGATION_FUNCTIONS,
        "order_price_diff": AGGREGATION_FUNCTIONS,
        "order_total_diff": AGGREGATION_FUNCTIONS,
    }

    df_agg = order_df.groupby(["customer_id"]).agg(agg_dict)
    return df_agg, agg_dict


def feature_sparta(df): # 횟수 x, 비율 x, 첫 구매, 마지막 구매
    df["sparta_order_ts"] = df["order_date"].astype(np.int64) // 1e9

    by = df.groupby(['customer_id', 'order_date']).sum()
    by['sparta'] = (by["total"] > 300).astype(int)

    by = by[by['sparta'] > 0]

    agg_dict = {
        "sparta_order_ts": ["first", "last"],
        # "sparta": ["sum"],
    }

    by = by.groupby(['customer_id']).agg(agg_dict)

    return by, agg_dict
    # return result
    


def feature_engineering(df, year_month):
    df = df.copy()

    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime("%Y-%m")

    # train, test 데이터 선택
    train = df[df["order_date"] < prev_ym]
    test = df[df["order_date"] < year_month]

    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[["customer_id", "year_month", "label"]]
    test_label = generate_label(df, year_month)[["customer_id", "year_month", "label"]]

    train_agg = pd.DataFrame()
    test_agg = pd.DataFrame()

    #        Base  Cumsum  Nunique  TD1    TD2    TS    Sparta
    # button = [True, False, False, False, False, False, False]
    button = [True, True, True, True, False, True, False]
    # button = [True] * 7
    feature_collect = []


    # 1 Base aggregations: 0.8100
    if button[0]:
        base_train_agg = feature_agg(train)
        base_test_agg = feature_agg(test)

        cols = new_cols(base_train_agg)

        base_train_agg.columns = cols
        base_test_agg.columns = cols

        # base_train_agg.drop(columns=['price-sum', 'price-max', 'price-min'], inplace = True)
        # base_train_agg.drop(columns=['quantity-sum', 'quantity-max', 'quantity-min'], inplace = True)
        # base_test_agg.drop(columns=['price-sum', 'price-max', 'price-min'], inplace = True)
        # base_test_agg.drop(columns=['quantity-sum', 'quantity-max', 'quantity-min'], inplace = True)

        feature_collect.append([base_train_agg, base_test_agg])


    # 2 Cumsum aggregations: 0.8383
    if button[1]:
        cumsum_train_agg = feature_cumsum(train)
        cumsum_test_agg = feature_cumsum(test)

        cols = new_cols(cumsum_train_agg)

        cumsum_train_agg.columns = cols
        cumsum_test_agg.columns = cols
        feature_collect.append([cumsum_train_agg, cumsum_test_agg])


    # 3 Nunique aggregations: 0.7930
    if button[2]:
        cols_select = ["order_id", "product_id"]

        nunique_train_agg = train.groupby(["customer_id"])[cols_select].agg(["nunique"])
        nunique_test_agg = test.groupby(["customer_id"])[cols_select].agg(["nunique"])

        cols = new_cols(nunique_train_agg)
        nunique_train_agg.columns = cols
        nunique_test_agg.columns = cols
        feature_collect.append([nunique_train_agg, nunique_test_agg])


    # 4-1 Time Series diff feature generation: 0.8510
    if button[3]:
        train, test = df_diff(df, prev_ym, year_month)
        diff_train_agg, agg_dict = feature_time_diff(train)
        diff_test_agg, agg_dict = feature_time_diff(test)

        cols = new_cols_dict(agg_dict)

        diff_train_agg.columns = cols
        diff_test_agg.columns = cols

        diff_train_agg.drop(columns=['order_ts_diff-min'], inplace = True)
        diff_test_agg.drop(columns=['order_ts_diff-min'], inplace = True)

        feature_collect.append([diff_train_agg, diff_test_agg])


    # 4-2 Time Series order diff feature generation:
    if button[4]:
        order_diff_train_agg, agg_dict = order_diff(train)
        order_diff_test_agg, agg_dict = order_diff(test)

        cols = new_cols_dict(agg_dict)

        order_diff_train_agg.columns = cols
        order_diff_test_agg.columns = cols
        feature_collect.append([order_diff_train_agg, order_diff_test_agg])


    # 5 Time Series aggregations: 0.6677
    if button[5]:
        # 추가적인 데이터 처리 - month, year_month 생성
        train["month"] = train["order_date"].dt.month
        train["year_month"] = train["order_date"].dt.strftime("%Y-%m")

        test["month"] = test["order_date"].dt.month
        test["year_month"] = test["order_date"].dt.strftime("%Y-%m")

        # Aggregations
        cols_select = ["month", "year_month"]

        ts_train_agg = train.groupby(["customer_id"])[cols_select].agg(
            [lambda x: x.value_counts().index[0]]
        )
        ts_test_agg = test.groupby(["customer_id"])[cols_select].agg(
            [lambda x: x.value_counts().index[0]]
        )

        ts_train_agg.columns = ["month-mode", "year_month-mode"]
        ts_test_agg.columns = ["month-mode", "year_month-mode"]
        feature_collect.append([ts_train_agg, ts_test_agg])


    # 6 Sparta!
    if button[6]:
        sparta_train, agg_dict = feature_sparta(train)
        sparta_test, agg_dict = feature_sparta(test)
        
        cols_select = new_cols_dict(agg_dict)

        sparta_train.columns = cols_select
        sparta_test.columns = cols_select

        feature_collect.append([sparta_train, sparta_test])


    # feature merge party
    for train, test in feature_collect:
        if train_agg.empty:
            train_agg = train
            test_agg = test
            continue

        train_agg = train.merge(train_agg, on=["customer_id"], how="left")
        test_agg = test.merge(test_agg, on=["customer_id"], how="left")


    # Label merge
    train_set = train_label.merge(train_agg, on=["customer_id"], how="left")
    test_set = test_label.merge(test_agg, on=["customer_id"], how="left")

    features = train_set.drop(columns=["customer_id", "label", "year_month"]).columns


    # year_month-mode: 2010-06 -> 6.0
    train_set, test_set = feature_preprocessing(train_set, test_set, features)
    print("train_set.shape", train_set.shape, ", test_set.shape", test_set.shape)

    return train_set, test_set, train_set["label"], features


if __name__ == "__main__":
    # 데이터 파일 읽기
    df = pd.read_csv("/opt/ml/code/input/train.csv", parse_dates=["order_date"])

    # 예측할 연월 설정
    year_month = "2011-12"
    # feature_engineering(df, year_month)

    x_train, test_setst, all_train, features = feature_engineering(df, year_month)
    print(f"x_train:{x_train}")
    print(f"test_setst:{test_setst}")
    print(f"all_train:{all_train}")
    print(f"features:{features}")

    x_train.to_csv('/opt/ml/code/notebook/data/check.csv')
    # test_setst.to_csv('/opt/ml/code/notebook/data/test.csv')
    # print(generate_label(data, year_month))
    # for f in feature_engineering1(data, year_month):
    #     print(f)
