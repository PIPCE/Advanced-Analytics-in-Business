import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from ..boost import Xgbst, CSXgboost, cv_csxgboost, XgbstFeatureImportance
from ..design_and_metrics import  PerformanceMetrics
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lime
import lime.lime_tabular as lt


def feature_imp(methods):

        feature_imp_dict = XgbstFeatureImportance.get_feature_importance_xgbst(methods)
        keys = list(feature_imp_dict.keys())
        vals = list(feature_imp_dict.values())

        feat_imp = pd.DataFrame({'feature' :keys, 'importance': vals})
        feat_imp = feat_imp.sort_values('importance')

        ax = sns.barplot(x=feat_imp['feature'], y=feat_imp['importance'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",fontsize=7)
        plt.title('Feature Importance')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()

