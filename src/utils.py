import os
import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import wandb

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# 시드 고정 함수
def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


# 평가 지표 출력 함수
def print_score(label, pred, WANDB_USE, prob_thres=0.5):
    print("Precision: {:.5f}".format(precision_score(label, pred > prob_thres)))
    print("Recall: {:.5f}".format(recall_score(label, pred > prob_thres)))
    print("F1 Score: {:.5f}".format(f1_score(label, pred > prob_thres)))
    print("ROC AUC Score: {:.5f}".format(roc_auc_score(label, pred)))

    if WANDB_USE:
        wandb.log({
            "ROC AUC Score": roc_auc_score(label, pred),
            "Precision": precision_score(label, pred > prob_thres), 
            "Recall": recall_score(label, pred > prob_thres),
            "F1 Score": f1_score(label, pred > prob_thres),
            })

    # score = [
    #     roc_auc_score(label, pred),
    #     precision_score(label, pred > prob_thres),
    #     recall_score(label, pred > prob_thres),
    #     f1_score(label, pred > prob_thres),
    #     ]

    # colors = sns.color_palette('hls',len(label))
    # label = ["ROC AUC Score", "Precision", "Recall", "F1 Score"]

    # plt.xticks(range(len(label)), label, fontsize=15)
    # plt.bar(
    #     x = [0, 1, 2, 3], 
    #     height = score, 
    #     width = 0.5,
    #     bottom = None,
    #     color = colors
    #     )

    # wandb.log({"Score chart": plt})


def plot_feature_importances(df, color='dodgerblue', figsize=(40, 100)):
    # 피처 중요도 순으로 내림차순 정렬
    df = df.sort_values('importance', ascending = False).reset_index(drop = True)
    
    # 피처 중요도 정규화 및 누적 중요도 계산
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    plt.style.use('fivethirtyeight')
    # 피처 중요도 순으로 n개까지 바플롯으로 그리기
    df.plot.barh(y='importance_normalized', 
                            x='feature', color=color, 
                            edgecolor='k', figsize=figsize,
                            legend=False)

    plt.xlabel('Normalized Importance', size=18); plt.ylabel(''); 
    plt.title(f'Important Features Ratio', size=18)
    plt.gca().invert_yaxis()
    # wandb.log({"Important Features chart": plt})


def plot_roc_curve(label, predict):
    fpr, tpr, thresholds = roc_curve(label, predict)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    wandb.log({"Plot ROC curve": plt})