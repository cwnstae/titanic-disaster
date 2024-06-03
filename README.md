[Visit to see my portfolio [Link]](https://cwnstae.github.io/data-analytic-portfolio/).

# Machine Learning - Titanic Disaster
In this project, I will use the dataset from the "Titanic: Machine Learning from Disaster" competition on Kaggle. I will demonstrate my approach to analyzing the features and optimizing my model to predict who will survive this disaster.

# The Challange ([Link](https://www.kaggle.com/competitions/titanic))
The sinking of the Titanic is one of the most infamous shipwrecks in history.
On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg.

Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

# Dataset
Overview
The data has been split into two groups:

training set (train.csv)
test set (test.csv)
The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.

<table>
<tbody>
<tr><th><b>Variable</b></th><th><b>Definition</b></th><th><b>Key</b></th></tr>
<tr>
<td>survival</td>
<td>Survival</td>
<td>0 = No, 1 = Yes</td>
</tr>
<tr>
<td>pclass</td>
<td>Ticket class</td>
<td>1 = 1st, 2 = 2nd, 3 = 3rd</td>
</tr>
<tr>
<td>sex</td>
<td>Sex</td>
<td></td>
</tr>
<tr>
<td>Age</td>
<td>Age in years</td>
<td></td>
</tr>
<tr>
<td>sibsp</td>
<td># of siblings / spouses aboard the Titanic</td>
<td></td>
</tr>
<tr>
<td>parch</td>
<td># of parents / children aboard the Titanic</td>
<td></td>
</tr>
<tr>
<td>ticket</td>
<td>Ticket number</td>
<td></td>
</tr>
<tr>
<td>fare</td>
<td>Passenger fare</td>
<td></td>
</tr>
<tr>
<td>cabin</td>
<td>Cabin number</td>
<td></td>
</tr>
<tr>
<td>embarked</td>
<td>Port of Embarkation</td>
<td>C = Cherbourg, Q = Queenstown, S = Southampton</td>
</tr>
</tbody>
</table>

# Feature Engineering
## 1. Dealing with Missing Values

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Pclass       891 non-null    int64  
 2   Name         891 non-null    object 
 3   Sex          891 non-null    object 
 4   Age          714 non-null    float64
 5   SibSp        891 non-null    int64  
 6   Parch        891 non-null    int64  
 7   Ticket       891 non-null    object 
 8   Fare         891 non-null    float64
 9   Cabin        204 non-null    object 
 10  Embarked     889 non-null    object 
dtypes: float64(2), int64(4), object(5)
memory usage: 76.7+ KB
```
<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>

You may notice that the age feature has approximately 20% missing values. Despite this, the data is crucial for predicting survival. Therefore, I will use a technique called "Imputation" to fill in the missing values instead of dropping the entire column. Conversely, due to the substantial missing data in the Cain feature (80%), I have decided not to utilize this column.
![image](https://github.com/cwnstae/titanic-disaster/assets/24621204/8f4f4305-e889-4777-acaf-e03bd3714f41)

```python
# Imputation
my_imputer = SimpleImputer()
imputed_X_train = X_train.copy()
imputed_X_test = X_test.copy()

imputed_age_train = pd.DataFrame(my_imputer.fit_transform(X_train[["Age"]]))
imputed_age_test = pd.DataFrame(my_imputer.transform(X_test[["Age"]]))

# Put back the column names
imputed_age_train.columns = X_train[["Age"]].columns
imputed_age_test.columns = X_test[["Age"]].columns

imputed_X_train["Age"] = imputed_age_train["Age"]
imputed_X_test["Age"] = imputed_age_test["Age"]

imputed_X_train.drop(['Cabin', 'PassengerId'],axis='columns', inplace=True)
imputed_X_test.drop(['Cabin', 'PassengerId'],axis='columns', inplace=True)

imputed_X_train.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 9 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   Pclass    891 non-null    int64  
 1   Name      891 non-null    object 
 2   Sex       891 non-null    object 
 3   Age       891 non-null    float64
 4   SibSp     891 non-null    int64  
 5   Parch     891 non-null    int64  
 6   Ticket    891 non-null    object 
 7   Fare      891 non-null    float64
 8   Embarked  889 non-null    object 
dtypes: float64(2), int64(3), object(4)
memory usage: 62.8+ KB

```

## 2. Dealing with Categorical Variables
For the categorical variables, I will use a technique called "ordinal_encoder" for the Neighborhood variable, which has no more than 10 unique values.

![image](https://github.com/cwnstae/titanic-disaster/assets/24621204/2e36c487-763f-4036-be64-624cdc259aeb)


```python
categorical_cols = [cname for cname in imputed_X_train.columns if
                    imputed_X_train[cname].nunique() < 10 and 
                    imputed_X_train[cname].dtype == "object"]
numerical_cols = [cname for cname in imputed_X_train.columns if 
                imputed_X_train[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train_Featured = imputed_X_train[my_cols].copy()
X_test_Featured = imputed_X_test[my_cols].copy()


# Apply ordinal encoder 
ordinal_encoder = OrdinalEncoder() # Your code here
X_train_Featured[categorical_cols] = ordinal_encoder.fit_transform(X_train_Featured[categorical_cols])
X_test_Featured[categorical_cols] = ordinal_encoder.fit_transform(X_test_Featured[categorical_cols])


# Imputation
my_imputer = SimpleImputer()
X_train_Featured_imputed = pd.DataFrame(my_imputer.fit_transform(X_train_Featured))
X_test_Featured_imputed = pd.DataFrame(my_imputer.transform(X_test_Featured))

# Imputation removed column names; put them back
X_train_Featured_imputed.columns = X_train_Featured.columns
X_test_Featured_imputed.columns = X_train_Featured.columns

X_train_Featured_imputed.info()
X_test_Featured_imputed.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   Sex       891 non-null    float64
 1   Embarked  891 non-null    float64
 2   Pclass    891 non-null    float64
 3   Age       891 non-null    float64
 4   SibSp     891 non-null    float64
 5   Parch     891 non-null    float64
 6   Fare      891 non-null    float64
dtypes: float64(7)
memory usage: 48.9 KB
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   Sex       418 non-null    float64
 1   Embarked  418 non-null    float64
 2   Pclass    418 non-null    float64
 3   Age       418 non-null    float64
 4   SibSp     418 non-null    float64
 5   Parch     418 non-null    float64
 6   Fare      418 non-null    float64
dtypes: float64(7)
memory usage: 23.0 KB


```
## 3. Feature Score
Before diving into the machine learning, let's explore how features (columns) affect survival. Who is likely to survive the disaster? I will use the Mutual Information (MI) score to assess the impact of each feature on survival.

```python
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

discrete_features = X_train_Featured_imputed.dtypes == int
mi_scores = make_mi_scores(X_train_Featured_imputed, y_train, discrete_features)
print (mi_scores)
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
```

![image](https://github.com/cwnstae/titanic-disaster/assets/24621204/2c74fdcf-82ae-4007-8b58-b396244af4f4)

It looks like `Sex` and `Fare` and `Pclass` is the most influence let's plot and explain

![image](https://github.com/cwnstae/titanic-disaster/assets/24621204/cfdfa1f5-92bf-4bb3-83d3-c1fbf7e2600e)

![image](https://github.com/cwnstae/titanic-disaster/assets/24621204/c9c31190-a831-42be-bdb7-e26e435baf15)

![image](https://github.com/cwnstae/titanic-disaster/assets/24621204/35a1213d-7516-434d-be8a-6ded7d8de0fe)

- Summary
 - Women are more likely to survive than men.
 - Wealthier individuals are more likely to survive.
 - Passengers in higher classes (with 1st class being the highest) are more likely to survive.

## 4. Training Model
I will use binary classification deep learning to create the model.
```python
input_shape = [X_train_Featured_imputed.shape[1]]
input_shape
model = keras.Sequential([
        layers.BatchNormalization(input_shape=input_shape),
        layers.Dense(16, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.5),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.5),
        layers.Dense(1, activation='sigmoid')
    ]
    )
model.compile(
    optimizer = 'adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    patience=100,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    X_train_Featured_imputed, y_train,
    validation_data=(X_test_Featured_imputed, y_test),
    batch_size=256,
    epochs=4000,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")

y_pred = (model.predict(X_test_Featured_imputed) >= 0.5).astype(int)
print('Validation Accuracy:', accuracy_score(y_test, y_pred))
```
`Validation Accuracy: 0.930622009569378`

![image](https://github.com/cwnstae/titanic-disaster/assets/24621204/f876b3fc-86d9-45f6-a007-47d7b5dae548)

I created a binary classification model with 93% accuracy, look at the cross-entropy graph show that `val_loss` is better than `loss`. it can indicate a few possible scenarios
 - Dropout or Regularization Effects.
 - Data Differences.
 - Batch Normalization effect.
