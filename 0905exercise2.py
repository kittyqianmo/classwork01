

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')



pd.set_option('display.max_rows', 1000)


df_test=pd.read_csv("/kaggle/input/competitions/sml-classwork-02-2026/test.csv")

df_test.head()
df_test.columns.tolist()
#print(df_test.isna().any())

for i in range(64,244):
    for j in range(df_test.shape[0]):
        if pd.isna(df_test.iloc[j,i])==True or df_test.iloc[j,i]==0:
            df_test.iloc[j,i]=df_test.iloc[:,i].mean()


for i in range(0,64):
    for j in range(df_test.shape[0]):
        if df_test.iloc[j,i]==0:
            df_test.iloc[j,i]=df_test.iloc[:,i].mean()

#print(df_test.isna().any())
has_zero = (df_test == 0).any().any()


def decodingmonth(tempdf):
    monthmap={"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    tempdf['month_num'] = tempdf['month'].map(monthmap)
    tempdf['month_sin'] = np.sin(2 * np.pi * tempdf['month_num'] / 12)
    tempdf['month_cos'] = np.cos(2 * np.pi * tempdf['month_num'] / 12)


def decodingday(tempdf):
    daymap={"Morning":1,"Afternoon":2,"Evening":3,"Night":4}
    tempdf['day_num'] = tempdf['day'].map(daymap)
    tempdf['day_sin'] = np.sin(2 * np.pi * tempdf['day_num'] / 4)
    tempdf['day_cos'] = np.cos(2 * np.pi * tempdf['day_num'] / 4)

decodingmonth(df_test)
decodingday(df_test)

df_train=pd.read_csv("/kaggle/input/competitions/sml-classwork-02-2026/train.csv")


for i in range(0,244):
    for j in range(df_train.shape[0]):
        if pd.isna(df_train.iloc[j,i])==True or df_train.iloc[j,i]==0:
            df_train.iloc[j,i]=df_train.iloc[:,i].mean()


#print(df_train.isna().any())
has_zero = (df_train == 0).any().any()
#print(has_zero)

decodingmonth(df_train)
decodingday(df_train)

df_train.head()

def fit_row(row,x):
    y = row.to_numpy(dtype=float)
    k, b = np.polyfit(x, y, 1)
    return pd.Series({'slope': k, 'intercept': b})

def get_attributes(a,b,dftemp,prefix):
    cols = dftemp.columns[a:b]
    dftemp[prefix+'_mean']= dftemp[cols].mean(axis=1)
    dftemp[prefix+'_sd']=dftemp[cols].std(axis=1)
    dftemp[prefix+'_min']=dftemp[cols].min(axis=1)
    dftemp[prefix+'_max']=dftemp[cols].max(axis=1)
    dftemp[prefix+'_range']=dftemp[prefix+'_max']-dftemp[prefix+'_min']
    sub=dftemp.iloc[:, a:b]
    x = np.arange(sub.shape[1])
    dftemp[[prefix+'_slope', prefix+'_intercept']] = sub.apply(fit_row, axis=1,args=(x,))


get_attributes(5,65,df_train,'sp')
get_attributes(65,125,df_train,'cs')
get_attributes(125,185,df_train,'fcd')
get_attributes(185,245,df_train,'al')

get_attributes(4,64,df_test,'sp')
get_attributes(64,124,df_test,'cs')
get_attributes(124,184,df_test,'fcd')
get_attributes(184,244,df_test,'al')

df_train = df_train.drop(columns=df_train.columns[5:245])
df_train = df_train.drop(columns=['month', 'day', 'month_num', 'day_num'])
df_test=df_test.drop(columns=df_test.columns[4:244])
df_test = df_test.drop(columns=['month', 'day', 'month_num', 'day_num'])
df_test.head()

df_train.head()

df_train.shape

df_test.shape

y_label=df_train.pop('hr')
X_label=df_train

X_new=df_test

n_features=len(X_label.columns)
gammaTest = 1.0 / (n_features)
print(gammaTest)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin


X_train, X_val, y_train, y_val = train_test_split(
    X_label, y_label, test_size=0.2, random_state=42
)



#Model 1 : RandomFeatures ridge regression

pipe = make_pipeline(
    StandardScaler(),
    RBFSampler(random_state=42),
    Ridge(),
)

param_grid = {
    "rbfsampler__gamma": [0.01,0.03, 0.1, 1.0],
    "rbfsampler__n_components": [500, 1000,2000],
    "ridge__alpha": [1e-3, 1e-2,1e-1,1],
}


grid = GridSearchCV(pipe, param_grid, cv=5,
                    scoring="neg_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)


y_val_pred = grid.predict(X_val)
print("Test MSE:", mean_squared_error(y_val, y_val_pred))


best_params = grid.best_params_
final_model = make_pipeline(
    StandardScaler(),
    RBFSampler(gamma=best_params["rbfsampler__gamma"],
               n_components=best_params["rbfsampler__n_components"],
               random_state=42),
    Ridge(alpha=best_params["ridge__alpha"]),
)
final_model.fit(X_label, y_label)


y_new_pred = final_model.predict(X_new)


#Model 2: NTK

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


def ntk_relu(X1, X2, depths):
    depths = set(depths)
    d = X1.shape[1]
    s1 = np.linalg.norm(X1, axis=1)
    s2 = np.linalg.norm(X2, axis=1)
    scale = np.outer(s1, s2) / d

    A = X1 / np.maximum(s1, 1e-12)[:, None]
    B = X2 / np.maximum(s2, 1e-12)[:, None]
    r = A @ B.T
    t = r
    out = {1: scale * t} if 1 in depths else {}

    for L in range(2, max(depths) + 1):
        r = np.clip(r, -1.0, 1.0)
        theta = np.arccos(r)
        r_dot = (np.pi - theta) / np.pi
        r = (np.sqrt(1.0 - r * r) + (np.pi - theta) * r) / np.pi
        t = r + t * r_dot
        if L in depths:
            out[L] = scale * t
    return out


depths = [2, 3, 4, 5]
alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

scaler = StandardScaler().fit(X_train)
Xtr, Xva = scaler.transform(X_train), scaler.transform(X_val)
m = np.mean(y_train)
ytr = np.ravel(y_train) - m

K_tr = ntk_relu(Xtr, Xtr, depths)
K_va = ntk_relu(Xva, Xtr, depths)

best = dict(score=-np.inf)
for L in depths:
    gs = GridSearchCV(KernelRidge(kernel="precomputed"), {"alpha": alphas},
                      cv=5, scoring="neg_mean_squared_error", n_jobs=1)
    gs.fit(K_tr[L], ytr)
    if gs.best_score_ > best["score"]:
        best = dict(score=gs.best_score_, depth=L,
                    alpha=gs.best_params_["alpha"], model=gs.best_estimator_)

y_val_pred = best["model"].predict(K_va[best["depth"]]) + m
print("Validation MSE:", mean_squared_error(y_val, y_val_pred))
print("Best:", best["depth"], best["alpha"])


L = best["depth"]
scaler = StandardScaler().fit(X_label)
Xl, Xn = scaler.transform(X_label), scaler.transform(X_new)
m = np.mean(y_label)
K_ll = ntk_relu(Xl, Xl, [L])[L]
K_nl = ntk_relu(Xn, Xl, [L])[L]
final = KernelRidge(kernel="precomputed", alpha=best["alpha"]).fit(K_ll, np.ravel(y_label) - m)
y_new_pred = final.predict(K_nl) + m

#Model 3: catboost

from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

cat_model = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=42,
    thread_count=1,
    iterations=2000,
    early_stopping_rounds=50,
    boosting_type="Plain",
    subsample=0.8,
    colsample_bylevel=0.8,
    l2_leaf_reg=3.0,
    verbose=False,
)

param_grid_cat = {
    "depth": [8],
    "learning_rate": [0.05],
    "l2_leaf_reg": [3.0],
    "min_data_in_leaf": [1]
}

grid_cat = GridSearchCV(
    cat_model, param_grid_cat,
    cv=5, scoring="neg_mean_squared_error",
    n_jobs=-1, verbose=2,
)

grid_cat.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
)

print("Best params:", grid_cat.best_params_)
print("Best CV MSE:", -grid_cat.best_score_)

y_val_pred_cat = grid_cat.predict(X_val)
print("CatBoost Validation MSE:",
      mean_squared_error(y_val, y_val_pred_cat))

best_params=grid_cat.best_params_
best_model = grid_cat.best_estimator_
best_iter = best_model.get_best_iteration() if best_model.get_best_iteration() else None
print("best_iteration:", best_iter)


if best_iter and best_iter > 0:
    final_iter = int(best_iter * 1.1)
else:
    final_iter = tree_count
final_iter = max(final_iter, 100)


final_cat = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=42,
    thread_count=-1,
    iterations=final_iter,
    depth=best_params["depth"],
    learning_rate=best_params["learning_rate"],
    l2_leaf_reg=best_params["l2_leaf_reg"],
    subsample=0.8,
    colsample_bylevel=0.8,
    verbose=False,
    min_data_in_leaf=1
)

final_cat.fit(X_label, y_label)

y_new_pred_cat = final_cat.predict(X_new)

min(y_new_pred)

max(y_new_pred)

result = pd.DataFrame({
    'id': X_new['id'].values,
    'hr': y_new_pred
})

result.to_csv("result.csv", index=False)