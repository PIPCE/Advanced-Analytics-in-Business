import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap


from ..boost import Xgbst, CSXgboost, cv_csxgboost, XgbstFeatureImportance
from ..design_and_metrics import  PerformanceMetrics
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lime
import lime.lime_tabular as lt


def feature_imp_impurity(method, string):

        feature_imp_dict = XgbstFeatureImportance.get_feature_importance_xgbst(method)
        keys = list(feature_imp_dict.keys())
        vals = list(feature_imp_dict.values())

        feat_imp = pd.DataFrame({'feature' :keys, 'importance': vals})
        feat_imp = feat_imp.sort_values('importance')

        ax = sns.barplot(x=feat_imp['feature'], y=feat_imp['importance'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",fontsize=7)
        plt.title('Feature Importance')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(string+"Feature Importance.png")
        plt.close()

def feature_imp_shap(method, covariates, string):

        #https://github.com/slundberg/shap

        explainer = shap.Explainer(method)
        shap_values = explainer(covariates)
        shap.plots.beeswarm(shap_values)
        plt.title('Shap Values')
        plt.tight_layout()
        plt.savefig(string+"shap.png")
        plt.close()

        shap.summary_plot(shap_values, covariates, feature_names=list(covariates.columns.values))
        plt.title('Shap Values all')
        plt.tight_layout()
        plt.savefig(string+"shap.png")
        plt.close()
