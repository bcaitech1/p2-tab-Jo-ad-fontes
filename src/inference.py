# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta
import argparse

import wandb

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Custom library
from utils import seed_everything, print_score
from features import feature_engineering
from model import lgbm, lgbm_params, cat, cat_params, xgboost, xgb_params

TOTAL_THRES = 300  # 구매액 임계값
SEED = 42  # 랜덤 시드
seed_everything(SEED)  # 시드 고정


data_dir = "../input"  # os.environ['SM_CHANNEL_TRAIN']
model_dir = "../model"  # os.environ['SM_MODEL_DIR']
output_dir = "../output"  # os.environ['SM_OUTPUT_DATA_DIR']


def run(args):
    # 데이터 파일 읽기
    data = pd.read_csv(data_dir + "/train.csv", parse_dates=["order_date"])

    # 피처 엔지니어링 실행
    train, test, y, features = feature_engineering(data, args.YEAR_MONTH)

    if args.MODEL == "lgbm":
        # Cross Validation Out Of Fold로 LightGBM 모델 훈련 및 예측
        y_oof, test_preds = lgbm(train, y, test, features, lgbm_params(), WANDB_USE=True)

    if args.MODEL == "cat":
        y_oof, test_preds = cat(train, y, test, features, cat_params(), WANDB_USE=True)

    if args.MODEL == "xgb":
        y_oof, test_preds = xgboost(train, y, test, features, xgb_params(), WANDB_USE=True)

    # 테스트 결과 제출 파일 읽기
    sub = pd.read_csv(data_dir + "/sample_submission.csv")

    # 테스트 예측 결과 저장
    sub["probability"] = test_preds

    # 제출 파일 쓰기
    os.makedirs(output_dir, exist_ok=True)
    sub.to_csv(
        os.path.join(output_dir, f"{args.MODEL}_{args.FILE_NAME}.csv"), index=False
    )


if __name__ == "__main__":
    # 인자 파서 선언
    parser = argparse.ArgumentParser()

    # baseline 모델 이름 인자로 받아서 model 변수에 저장
    parser.add_argument(
        "--MODEL", type=str, default="lgbm", help="set ML model name among lgbm"
    )
    parser.add_argument(
        "--YEAR_MONTH", type=str, default="2011-12", help="set predict year-month"
    )
    parser.add_argument("--FILE_NAME", type=str, default="test", help="set file name")

    args = parser.parse_args()

    # wandb
    wandb.init(project="stage-2-final", reinit=True)
    wandb.run.name = f'{args.MODEL}_{args.FILE_NAME}'
    

    print("Model:", args.MODEL)

    # train, test
    run(args)
