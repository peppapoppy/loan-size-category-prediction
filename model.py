import pandas as pd
import numpy as np
from xgboost import XGBClassifier


class Alpha:
    def __init__(self):

        self.cols = ['m1_long_term_debt',
             'm1_long_term_debt_per_total_capital',
             'm1_lt_debt_per_equity',
             'm1_total_debt',
             'm2_long_term_debt',
             'm1_loan__medium',
             'company_size__micro',
             'm1_total_debt_to_capital_pct',
             'm1_total_debt_per_total_liabilities_pct',
             'm1_total_liabilities_standard_or_utility_template',
             'm1_loan__no',
             'm2_long_term_debt_per_total_capital',
             'm1_total_debt_per_equity',
             'm2_total_debt',
             'm1_loan__large',
             'm1_total_assets',
             'm2_total_liabilities_standard_or_utility_template',
             'm2_lt_debt_per_equity',
             'company_size__small',
             'm1_loan__small',
             'm1_short_term_borrowings',
             'm3_long_term_debt',
             'm2_loan__no',
             'm1_total_capital',
             'm2_loan__medium',
             'm3_long_term_debt_per_total_capital',
             'm2_total_debt_to_capital_pct',
             'm2_total_assets',
             'm2_total_debt_per_total_liabilities_pct',
             'm3_other_non_current_liabilities',
             'm1_other_non_current_liabilities',
             'm1_other_non_current_liabilities_reported_private_only',
             'm3_total_liabilities_standard_or_utility_template',
             'm1_total_tangible_fixed_assets_reported_private_only',
             'm3_other_non_current_liabilities_reported_private_only',
             'm1_common_stock_total',
             'm1_retained_earnings',
             'm1_other_receivables',
             'company_size__medium',
             'm2_total_tangible_fixed_assets_reported_private_only',
             'm2_other_non_current_liabilities',
             'm2_loan__large',
             'm2_loan__small',
             'm1_current_assets_per_total_assets',
             'm1_total_equity',
             'm2_total_debt_per_equity',
             'm3_total_assets',
             'm2_other_non_current_liabilities_reported_private_only',
             'm2_short_term_borrowings',
             'm2_retained_earnings',
             'm3_total_tangible_fixed_assets_reported_private_only',
             'm2_total_capital',
             'm1_inventory',
             'm1_other_current_liabilities_reported_private_only',
             'm1_total_cash_and_short_term_investments',
             'm2_current_assets_per_total_assets',
             'm3_lt_debt_per_equity',
             'm1_long_term_leases',
             'm1_other_current_liabilities',
             'm1_current_ratio',
             'm1_loan__other',
             'm1_total_current_assets',
             'm2_total_equity',
             'm3_total_debt',
             'm2_provision_for_pension_reported_private_only',
             'm1_inventory_per_current_assets',
             'm1_current_income_taxes_payable',
             'm3_loan__no',
             'm2_total_current_assets',
             'm3_current_ratio'
             
             ]
        
    
    def preprocessing(self, df):
        
        cat_cols_ = ['m1_loan',
                     'm3_loan',
                     'zipcode',
                     'm2_loan',
                     'm1_is_pandemic',
                     'subindustry_level_3',
                     'city',
                     'state',
                     'company_status_type_name',
                     'm3_is_pandemic',
                     'm2_is_pandemic',
                     'company_size',
                     'subindustry_level_4',
                     'subindustry_level_2',
                     'num_employees', 
                     'masked_company_id', 
                     'year_founded']

        cat_cols = [
                     'm2_loan',
                     'm1_loan',
                     'm3_loan',
                     'm2_is_pandemic',
                     'm1_is_pandemic',
                     'm3_is_pandemic',
                     'company_status_type_name',
                     'company_size'
                     ]

        df_cat = df[cat_cols].copy()
        df = df.drop(cat_cols_, axis=1)
        df_cat.m1_is_pandemic = df_cat.m1_is_pandemic.replace(False,1).fillna(0)
        df_cat.m2_is_pandemic = df_cat.m2_is_pandemic.replace(False,1).fillna(0)
        df_cat.m3_is_pandemic = df_cat.m3_is_pandemic.replace(False,1).fillna(0)
        df_cat_1 = df_cat[['m1_is_pandemic', 'm2_is_pandemic', 'm3_is_pandemic']].copy()

        df_cat.m1_loan = df_cat.m1_loan.fillna('other')
        df_cat.m2_loan = df_cat.m2_loan.fillna('other')
        df_cat.m3_loan = df_cat.m3_loan.fillna('other')

        dum_df_m1_loan = pd.get_dummies(df_cat[['m1_loan']], columns=["m1_loan"], prefix=["m1_loan_"] )
        dum_df_m2_loan = pd.get_dummies(df_cat[['m2_loan']], columns=["m2_loan"], prefix=["m2_loan_"] )
        dum_df_m3_loan = pd.get_dummies(df_cat[['m3_loan']], columns=["m3_loan"], prefix=["m3_loan_"] )
        dum_df_company_size = pd.get_dummies(df_cat[['company_size']], columns=["company_size"], prefix=["company_size_"] )
        dum_df_company_status_type_name = pd.get_dummies(df_cat[['company_status_type_name']], columns=["company_status_type_name"], prefix=["company_status_type_name_"] )
        df_cat_new = dum_df_m1_loan.join(dum_df_m2_loan).join(dum_df_m3_loan).join(dum_df_company_size).join(dum_df_company_status_type_name).join(df_cat_1)

        df = df.join(df_cat_new)
        df = df.fillna(df.mean())
         
        return df
    
    def macro_f1_score(self, y_pred, dtrain):
        y_true = dtrain.get_label()
        n_labels = 4
        total_f1 = 0.
        for i in range(n_labels):
            yt = y_true == i
            yp = y_pred == i

            tp = np.sum(yt & yp)

            tpfp = np.sum(yp)
            tpfn = np.sum(yt)
            if tpfp == 0:
                
                precision = 0.
            else:
                precision = tp / tpfp
            if tpfn == 0:
                
                recall = 0.
            else:
                recall = tp / tpfn

            if precision == 0. or recall == 0.:
                f1 = 0.
            else:
                f1 = 2 * precision * recall / (precision + recall)
            total_f1 += f1
        return 'f1_err', 1-(total_f1 / n_label)


    def fit(self):
        df = pd.read_parquet('files/brook_mvp_loan_simpleindustry_all_2018.parquet')
        df = self.preprocessing(df)
        df = df[self.cols + ['loan_target']]


        y = df.loan_target
        X = df[self.cols]
        del df

        xgb = XGBClassifier(
                 learning_rate =0.05,
                 n_estimators=500, 
                 max_depth=10, 
                 min_child_weight=1,
                 gamma=0.1,
                 subsample=0.8, 
                 colsample_bytree=0.8,
                 objective= 'multi:softmax',
                 seed=27)
        self.model = xgb.fit(X, y, eval_metric=self.macro_f1_score)

        return self.model
       


    def fun(self, data):
        return data[['0','1', '2','3']].idxmax()

    def predict_batch(self, test_df):
        
        df = self.preprocessing(test_df)
        
        new_cols = [x for x in self.cols if x not in df.columns]
        for i in new_cols:
            df[i]=0

        df = df[self.cols]
        y_prob = self.model.predict_proba(df)

        xx = pd.DataFrame()
        xx['0'] = y_prob[:,0]
        xx['1'] = y_prob[:,1]
        xx['2'] = y_prob[:,2]
        xx['3'] = y_prob[:,3]
        xx[['1', '2','3']] = xx[['1', '2','3']]*7  #4-6
        xx['new_pred'] = xx.apply(self.fun, axis=1)
        y = list(xx['new_pred'])
        return y

