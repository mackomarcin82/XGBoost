import pandas as pd
import urllib.request
import zipfile
import xgboost as xgb
from sklearn import base, pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder 
from feature_engine import encoding, imputation
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Any, Dict, Union

def hyperparameter_tuning(space: Dict[str, Union[float, int]],
X_train: pd.DataFrame, y_train: pd.Series,
X_test: pd.DataFrame, y_test: pd.Series,
early_stopping_rounds: int=50,
metric:callable=accuracy_score)-> Dict[str, Any]:
    
    int_vals = ['max_depth', 'reg_alpha']
    space = {k: (int(val) if k in int_vals else val)
             for k,val in space.items()}
    
    space['early_stopping_rounds'] = early_stopping_rounds
    model = xgb.XGBClassifier(**space)
    evaluation = [(X_train, y_train),
    (X_test, y_test)]
    model.fit(X_train, y_train,
    eval_set=evaluation,
    verbose=False)

    pred = model.predict(X_test)
    score = metric(y_test, pred)

    return {'loss':-score, 'status': STATUS_OK, 'model': model}

def extract_zip(src, dst, member_name):
    url = src
    fname = dst
    fin = urllib.request.urlopen(url)
    data = fin.read()
    with open(dst, mode='wb') as fout:
        fout.write(data)
    with zipfile.ZipFile(dst) as z:
        kag = pd.read_csv(z.open(member_name),low_memory=False)
        kag_questions = kag.iloc[0]
        raw = kag.iloc[1:]
        return raw


def topn(ser, n=5, default='other'):
    counts = ser.value_counts()
    return ser.where(ser.isin(counts.index[:n]), default)

def tweak_kag(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Tweak the Kaggle survey data and return a new DataFrame.
    This function takes a Pandas DataFrame containing Kaggle
    survey data as input and returns a new DataFrame. The
    modifications include extracting and transforming certain
    columns, renaming columns, and selecting a subset of columns.
    Parameters
    ----------
    df_ : pd.DataFrame
    The input DataFrame containing Kaggle survey data.
    Returns
    -------
    pd.DataFrame
    The new DataFrame with the modified and selected columns.
    """
    return (
        df_.assign(
            age=df_.Q2.str.slice(0,2).astype(int),
            education=df_.Q4.replace({'Master’s degree': 18,
                'Bachelor’s degree': 16,
                'Doctoral degree': 20,
                'Some college/university study without earning a bachelor’s degree': 13,
                'Professional degree': 19,
                'I prefer not to answer': None,
                'No formal education past high school': 12}),
            major=(df_.Q5
                .pipe(topn, n=3)
                .replace({
                    'Computer science (software engineering, etc.)': 'cs',
                    'Engineering (non-computer focused)': 'eng',
                    'Mathematics or statistics': 'stat'})
                ),
            years_exp=(df_.Q8.str.replace('+','', regex=False)
                .str.split('-', expand=True)
                .iloc[:,0]
                .astype(float)),
            compensation=(df_.Q9.str.replace('+','', regex=False)
                .str.replace(',','', regex=False)
                .str.replace('500000', '500', regex=False)
                .str.replace('I do not wish to disclose my approximate yearly compensation',
                '0', regex=False)
                .str.split('-', expand=True)
                .iloc[:,0]
                .fillna(0)
                .astype(int)
                .mul(1_000)
                ),
            python=df_.Q16_Part_1.fillna(0).replace('Python', 1),
            r=df_.Q16_Part_2.fillna(0).replace('R', 1),
            sql=df_.Q16_Part_3.fillna(0).replace('SQL', 1)
        )#assign
        .rename(columns=lambda col:col.replace(' ', '_'))
        .loc[:, 'Q1,Q3,age,education,major,years_exp,compensation,'
        'python,r,sql'.split(',')]
    )


class TweakKagTransformer(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, ycol=None):
        self.ycol = ycol

    def transform(self, X):
        return tweak_kag(X)

    def fit(self, X, y=None):
        return self
    

def get_rawX_y(df, y_col):
    raw = (df
    .query('Q3.isin(["United States of America", "China", "India"]) '
    'and Q6.isin(["Data Scientist", "Software Engineer"])')
    )
    return raw.drop(columns=[y_col]), raw[y_col]


## Create a pipeline
kag_pl = pipeline.Pipeline(
    [('tweak', TweakKagTransformer()),
    ('cat', encoding.OneHotEncoder(top_categories=5,
    variables=['Q1', 'Q3', 'major'])),
    ('num_impute', imputation.MeanMedianImputer(imputation_method='median',
                                                variables=['education', 'years_exp']))]
)