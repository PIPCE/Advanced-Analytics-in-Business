
class XgbstFeatureImportance:

    def get_feature_importance_xgbst(xgbst):
        return xgbst.get_score(importance_type='gain')
