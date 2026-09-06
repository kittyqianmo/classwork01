# -*- coding: utf-8 -*-


import pandas as pd
df=pd.read_csv("/kaggle/input/competitions/sml-classwork-01-2026/test.csv")

print(df.head())
len(df)

print(pd.api.types.is_numeric_dtype(df['id']))
print(pd.api.types.is_numeric_dtype(df['age']))

print(df['younger_sib'].value_counts())
print(df['gender'].value_counts())
print(df['older_sib'].value_counts())
print(df['cat_dog'].value_counts())
print(df['music_train'].value_counts())
print(df['music_instru'].value_counts())
print(df['quite_now'].value_counts())
print(df['green_now'].value_counts())
print(df['quite_then'].value_counts())
print(df['green_then'].value_counts())

dictmapfortest={
    'I have both, 6 cats and 6 dogs (And other farm animals)':'Both',
    'Love them both':'Both',

}


df['cat_dog']=df['cat_dog'].replace(dictmapfortest)

print(df['cat_dog'].value_counts())


df.loc[:, 'music_instru'] = df['music_instru'].replace({
    'Dj console (? Is it an instrument? :,)': 'Dj console','Nope, none of them':'Nope'
})

print(df['music_instru'].value_counts())



df_train=pd.read_csv("/kaggle/input/competitions/sml-classwork-01-2026/train.csv")
print(df_train.head())

print(df_train['age'].value_counts())
print(df_train['younger_sib'].value_counts())
print(df_train['gender'].value_counts())
print(df_train['older_sib'].value_counts())
print(df_train['cat_dog'].value_counts())
print(df_train['music_train'].value_counts())
print(df_train['music_instru'].value_counts())
print(df_train['quite_now'].value_counts())
print(df_train['green_now'].value_counts())
print(df_train['quite_then'].value_counts())
print(df_train['green_then'].value_counts())

df_train=df_train[df_train['gender']!='apache elicopter']

dictmapfortrain={
    'both':'Both',
    'I like them, but not really a fan of furry animals':'Furry stuff? No thx...',
    'both!':'Both',
    'I like both':'Both',
    'I have both, 6 cats and 6 dogs (And other farm animals)':'Both',
    'Love them both':'Both',
    'Love both!!! ':'Both',
    'Both/none':'Both'

}


df_train['cat_dog']=df_train['cat_dog'].replace(dictmapfortrain)

value_counts_result=df_train['cat_dog'].value_counts().to_dict()
list1=[]
for key,value in value_counts_result.items():
    if value<2:
        list1.append(key)

df_train_updated1 = df_train[~df_train['cat_dog'].isin(list1)]
df_train_updated1['cat_dog'].value_counts()

df_train_updated1.loc[:, 'music_instru'] = df_train_updated1['music_instru'].replace({
    'Dj console (is an instrument? :,))': 'Dj console','Nope, none of them':'Nope'
})
df_train_updated2 = df_train_updated1[~df_train_updated1['music_instru'].isin(['Guitar, Baglama','Flute','Vocal chords','Guitar, flauto dolce ','Trumpet','Piano/Keyboard, Guitar, Drums, sing '])]
df_train_updated2['music_instru'].value_counts()

all_labels = ['Piano/Keyboard', 'Nope', 'Guitar', 'Violin', 'Drums', 'Dj console']


def safe_multi_hot_encode(data_series, known_labels):
    for index, row in data_series.iterrows():
        items=row['music_instru'].split(', ')
        filtered = [item for item in items if item in known_labels]
        for label in known_labels:
            if label in filtered:
                data_series.at[index, label] = 1
            else:
                data_series.at[index, label] = 0
    return data_series

df_train_updated2_copy= df_train_updated2.copy()
df_copy= df.copy()

df_train_encoded = safe_multi_hot_encode(df_train_updated2_copy, all_labels)
df_test_encoded = safe_multi_hot_encode(df_copy, all_labels)

df_train_encoded.head()
df_test_encoded.head()

df_train_encoded.drop(columns=['music_instru'], inplace=True)
df_test_encoded.drop(columns=['music_instru'], inplace=True)





print(len(df_train_updated2_copy))
df_test_encoded.head()

df_train_encoded['gender'] = df_train_encoded['gender'].map({'Female': 0, 'Male': 1}).astype(int)
df_test_encoded['gender'] = df_test_encoded['gender'].map({'Female': 0, 'Male': 1}).astype(int)


df_train_encoded['gender'].value_counts(dropna=False)


df_test_encoded['gender'].value_counts(dropna=False)

train_dummies = pd.get_dummies(df_train_encoded['cat_dog'], prefix='cat&dog_',dtype=int)

train_cols = train_dummies.columns


test_dummies = pd.get_dummies(df_test_encoded['cat_dog'], prefix='cat&dog_',dtype=int)

test_dummies = test_dummies.reindex(columns=train_cols, fill_value=0)


df_train_encoded = df_train_encoded.drop(columns=['cat_dog'])
df_test_encoded = df_test_encoded.drop(columns=['cat_dog'])


df_train_encoded[train_dummies.columns] = train_dummies
df_test_encoded[test_dummies.columns] = test_dummies





df_train_encoded['music_train'] = df_train_encoded['music_train'].map({'No': 0, 'Yes, less than 3 years': 1,'Yes, between 3 and 6 years':2,'Yes, between 6 and 10 years':3,'Yes, more than 10 years':4}).astype(int)
df_train_encoded['music_train'].value_counts()


df_test_encoded['music_train'] = df_test_encoded['music_train'].map({'No': 0, 'Yes, less than 3 years': 1,'Yes, between 3 and 6 years':2,'Yes, between 6 and 10 years':3,'Yes, more than 10 years':4}).astype(int)
df_test_encoded['music_train'].value_counts()

df_train_encoded.head()
df_test_encoded.head()

train_mean = df_train_encoded['age'].mean()

df_test_encoded.fillna(train_mean, inplace=True)

df_train_encoded['target'].value_counts()
#len(df_test_encoded)

from sklearn.linear_model import LogisticRegression


model = LogisticRegression(
    multi_class='multinomial',
    penalty='l1',
    solver='saga',
    C=1.0,
    max_iter=1000
)

X_train = df_train_encoded.drop(columns=['target'])
model.fit(X_train, df_train_encoded['target'])  # y_train 里有 0,1,2,3 四个类别

#cols_with_nan = df_test_encoded.columns[df_test_encoded.isna().any()].tolist()
#print(cols_with_nan)

y_pred = model.predict(df_test_encoded)


y_pred_proba = model.predict_proba(df_test_encoded)

df_test_encoded['target']=y_pred

test_res=df_test_encoded[['id', 'target']].copy()
test_res['target']=test_res['target'].astype(str)

test_res.to_csv('submission.csv', index=False)
