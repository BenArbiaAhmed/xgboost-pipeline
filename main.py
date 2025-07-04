import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


#Change URL here
data = pd.read_csv('train.csv')

# Change target here
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1)


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)


#Change the number of unique values per categorical column here
num_unique_values_category = 20
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < num_unique_values_category and
                        X_train_full[cname].dtype == "object"]


numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]


my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()


numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])



model = XGBRegressor(n_estimators=1000, learning_rate=0.03)


my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])



def score_dataset(param_x_train, param_x_valid, param_y_train, param_y_valid, pipeline):
    pipeline.fit(param_x_train, param_y_train)
    predictions = pipeline.predict(param_x_valid)
    return predictions

score = score_dataset(X_train, X_valid, y_train, y_valid, my_pipeline)
print('MAE:', score)


#In case you want to apply cross-validation
# from sklearn.model_selection import cross_val_score
#
# scores = -1 * cross_val_score(my_pipeline, X, y,
#                               cv=5,
#                               scoring='neg_mean_absolute_error')
#
# print("MAE scores:\n", scores)
