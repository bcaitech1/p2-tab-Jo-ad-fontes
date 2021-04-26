import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta

import wandb

from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb

from catboost import CatBoostClassifier, Pool

from utils import seed_everything, print_score, plot_feature_importances, plot_roc_curve
from features import generate_label, feature_engineering

TOTAL_THRES = 300  # 구매액 임계값
SEED = 42  # 랜덤 시드
seed_everything(SEED)  # 시드 고정
FEEATURE_FILE_NAME = "select"
"""
    머신러닝 모델 없이 입력인자으로 받는 year_month의 이전 달 총 구매액을 구매 확률로 예측하는 베이스라인 모델
"""


def lgbm_params():
    model_params = {
        'learning_rate': 0.024,
        "objective": "binary",  # 이진 분류
        "boosting_type": "gbdt",  # dart 오래걸림 'rf', 'gbdt'
        "metric": "auc",  # 평가 지표 설정
        'num_leaves': 8,
        'max_bin': 198,
        'min_data_in_leaf': 28,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 4,
        "n_estimators": 97015,  # 트리 개수
        "early_stopping_rounds": 100,
        "seed": SEED,
        "verbose": -1,
        "n_jobs": -1,
    }

    return model_params


def cat_params():
    model_params = {
        'n_estimators': 10000, # 트리 개수
        'learning_rate': 0.09, # 학습률
        'eval_metric': 'AUC', # 평가 지표 설정
        'loss_function': 'Logloss', # 손실 함수 설정
        'random_seed': SEED,
        'metric_period': 100,
        'od_wait': 100, # early stopping round
        'depth': 7, # 트리 최고 깊이
        'rsm': 0.66, # 피처 샘플링 비율
        'boosting_type': 'Ordered',
        'bootstrap_type': 'MVS'
    }

    return model_params


def xgb_params():
    model_params = {
        'objective': 'binary:logistic', # 이진 분류
        'learning_rate': 0.02211903224420238, # 학습률
        'max_depth': 6, # 트리 최고 깊이
        'colsample_bytree': 0.7173762029018879, # 피처 샘플링 비율
        'subsample': 0.8450778548079342, # 데이터 샘플링 비율
        'eval_metric': 'auc', # 평가 지표 설정
        'seed': SEED,
    }    

    return model_params


def lgbm(train, y, test, features, model_params, WANDB_USE, categorical_features="auto", folds=10):
    if WANDB_USE:
        wandb.config.update(model_params)

    x_train = train[features]
    x_test = test[features]

    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])

    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])

    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0

    # 피처 중요도를 저장할 데이터 프레임 선언
    feature_importance = pd.DataFrame()
    feature_importance["feature"] = features

    # Stratified K Fold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    # TimeSeriesSplit
    # skf = TimeSeriesSplit(n_splits=24)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]

        print(f"fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}")

        # LightGBM 데이터셋 선언
        dtrain = lgb.Dataset(x_tr, label=y_tr)
        dvalid = lgb.Dataset(x_val, label=y_val)

        # LightGBM 모델 훈련

        clf = lgb.train(
            model_params,
            dtrain,
            valid_sets=[dtrain, dvalid],  # Validation 성능을 측정할 수 있도록 설정
            categorical_feature=categorical_features,
            verbose_eval=200,
        )

        # Validation 데이터 예측
        val_preds = clf.predict(x_val)

        # Validation index에 예측값 저장
        y_oof[val_idx] = val_preds

        # 폴드별 Validation 스코어 측정
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print("-" * 80)
        if WANDB_USE:
            wandb.log({"AUC": roc_auc_score(y_val, val_preds)})

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds

        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(x_test) / folds

        # 폴드별 피처 중요도 저장
        feature_importance[f"fold_{fold+1}"] = clf.feature_importance()

        del x_tr, x_val, y_tr, y_val
        gc.collect()

    # 폴드별 Validation 스코어 출력 & Out Of Fold Validation 스코어 출력
    print(f"\nMean AUC = {score}")  
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}")  

    # 평가 지표 출력 함수
    print_score(y, y_oof, WANDB_USE)

    # 폴드별 피처 중요도 평균값 계산해서 저장
    fi_cols = [col for col in feature_importance.columns if "fold_" in col]
    feature_importance["importance"] = feature_importance[fi_cols].mean(axis=1)

    # feature 중요도 출력
    print(feature_importance) 
    feature_importance.to_csv(f'/opt/ml/code/output/fi_lgbm_{FEEATURE_FILE_NAME}.csv')
    # plot_feature_importances(feature_importance)
    # plot_roc_curve(y, y_oof)

    return y_oof, test_preds


def cat(train, y, test, features, model_params, WANDB_USE, categorical_features=None, folds=10):
    if WANDB_USE:
        wandb.config.update(model_params)

    x_train = train[features]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    feature_importance = pd.DataFrame()
    feature_importance['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')
        
        # CatBoost 모델 훈련
        clf = CatBoostClassifier(**model_params)
        clf.fit(x_tr, y_tr,
                eval_set=(x_val, y_val), # Validation 성능을 측정할 수 있도록 설정
                cat_features=categorical_features,
                use_best_model=True,
                verbose=True)
        
        # Validation 데이터 예측
        val_preds = clf.predict_proba(x_val)[:,1]
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 출력
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)
        if WANDB_USE:
            wandb.log({"AUC": roc_auc_score(y_val, val_preds)})

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict_proba(x_test)[:,1] / folds

        # 폴드별 피처 중요도 저장
        feature_importance[f'fold_{fold+1}'] = clf.feature_importances_
        
        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 평균 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력
    
    # 평가 지표 출력 함수
    print_score(y, y_oof, WANDB_USE)

    # 폴드별 피처 중요도 평균값 계산해서 저장
    fi_cols = [col for col in feature_importance.columns if 'fold_' in col]
    feature_importance['importance'] = feature_importance[fi_cols].mean(axis=1)
    feature_importance.to_csv(f'/opt/ml/code/output/fi_cat_{FEEATURE_FILE_NAME}.csv')

    return y_oof, test_preds


def xgboost(train, y, test, features, model_params, WANDB_USE, folds=10):
    if WANDB_USE:
        wandb.config.update(model_params)

    x_train = train[features]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    feature_importance = pd.DataFrame()
    feature_importance['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')
        
        # XGBoost 데이터셋 선언
        dtrain = xgb.DMatrix(x_tr, label=y_tr)
        dvalid = xgb.DMatrix(x_val, label=y_val)
        
        # XGBoost 모델 훈련
        clf = xgb.train(
            model_params,
            dtrain,
            num_boost_round=10000, # 트리 개수
            evals=[(dtrain, 'train'), (dvalid, 'valid')],  # Validation 성능을 측정할 수 있도록 설정
            verbose_eval=200,
            early_stopping_rounds=100
        )
        
        # Validation 데이터 예측
        val_preds = clf.predict(dvalid)
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 출력
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)
        if WANDB_USE:
            wandb.log({"AUC": roc_auc_score(y_val, val_preds)})

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(xgb.DMatrix(x_test)) / folds

        # 폴드별 피처 중요도 저장
        fi_tmp = pd.DataFrame.from_records([clf.get_score()]).T.reset_index()
        fi_tmp.columns = ['feature',f'fold_{fold+1}']
        feature_importance = pd.merge(feature_importance, fi_tmp, on='feature')

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 평균 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력
    
    # 평가 지표 출력 함수
    print_score(y, y_oof, WANDB_USE)

    # 폴드별 피처 중요도 평균값 계산해서 저장
    fi_cols = [col for col in feature_importance.columns if 'fold_' in col]
    feature_importance['importance'] = feature_importance[fi_cols].mean(axis=1)
    feature_importance.to_csv(f'/opt/ml/code/output/fi_xgb_{FEEATURE_FILE_NAME}.csv')

    return y_oof, test_preds