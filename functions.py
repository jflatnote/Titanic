import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import functions as funk

import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import norm

from sklearn import linear_model
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error

from mlxtend.regressor import StackingCVRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor

#Function to print a summary of a DataFrame
def preview_data(DataFrame):
    """[summary]

    Args:
        DataFrame ([type]): [description]
    """
        
    print("\n ----------Top-5- Record----------")
    print(DataFrame.head(5))

    print("\n -----------Information-----------")
    print(DataFrame.info())

    print("\n -----------Data Types-----------")
    print(DataFrame.dtypes)

    print("\n ----------Missing value-----------")
    print(DataFrame.isnull().sum())

    print("\n ----------Null value-----------")
    print(DataFrame.isna().sum())

    print("\n ----------Shape of Data----------")
    print(DataFrame.shape)

    print("\n ----------Number of duplicates----------")
    print('Number of duplicates:', len(DataFrame[DataFrame.duplicated()]))

# Function to calculate missing values by column
def missing_values_table(df):
    """[summary]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    
    # Total missing values
    mis_val = df.isnull().sum()

    #Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename( columns = {0:'Missing Values', 1:'% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

    # Print some summary of information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
    
    #Return the dataframe with missing information
    return mis_val_table_ren_columns

def encode_transform(app, col, le):
    """[summary]

    Args:
        app ([type]): [description]
        col ([type]): [description]
        le ([type]): [description]

    Returns:
        [type]: [description]
    """  
      
    # Transform both training and testing data
    app[col] = le[col].transform(app[col])
    return app, col, le

def get_model_results(model_name, val_y, preds):
    """[summary]

    Args:
        model_name ([type]): [description]
        val_y ([type]): [description]
        preds ([type]): [description]
    """   
     
    print(f'The accuracy of the {model_name} is {r2_score(val_y,preds)}')
    print(f'RMSE is : {mean_squared_error(val_y,preds)}')
    print(f'MAE is  : {mean_absolute_error(val_y,preds)}')

def run_linear_regression(train_X, train_y, val_X, val_y):
    """[summary]

    Args:
        train_X ([type]): [description]
        train_y ([type]): [description]
        val_X ([type]): [description]
        val_y ([type]): [description]

    Returns:
        [type]: [description]
    """    
    
    model_scores = pd.DataFrame(columns = ['model_name', 'run', 'run_notes', 'accuracy', 'RMSE', 'MAE'])
    
    # Create a linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(train_X, train_y)

    # Make predictions using the validation set
    y_pred = regr.predict(val_X)

    to_append = ['Linear Regression', 1, None, r2_score(val_y,y_pred), mean_squared_error(val_y,y_pred), 
                    mean_absolute_error(val_y,y_pred)]
    a_series = pd.Series(to_append, index = model_scores.columns)
    model_scores = model_scores.append(a_series, ignore_index=True)

    return y_pred, model_scores, regr

def run_xgboost(n_splits,train_X,train_y,val_X,val_y):
    """[summary]

    Args:
        n_splits ([type]): [description]
        train_X ([type]): [description]
        train_y ([type]): [description]
        val_X ([type]): [description]
        val_y ([type]): [description]

    Returns:
        [type]: [description]
    """  
    
    model_name = "XGBoost with GridSearchCV"
    model_scores = pd.DataFrame(columns = ['model_name', 'run', 'run_notes', 'accuracy', 'RMSE', 'MAE'])
    run=1  
        
    kfolds = KFold(n_splits, shuffle=False)
    
    estimator = xgb.XGBRegressor(objective='reg:squarederror', verbose=False)
    parameters = {
        'max_depth': range (2, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05],
        'gamma': range(0,20,1),
        #'lambda': range(0,1,.1)
    }

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        scoring=make_scorer(r2_score),
        n_jobs = 10,
        cv = kfolds,
        verbose=False
    )

    grid_search.fit(train_X, train_y)

    
    xg_model = grid_search.best_estimator_
    preds = xg_model.predict(val_X)

    to_append = [model_name, run, None, r2_score(val_y,preds), mean_squared_error(val_y,preds), 
                    mean_absolute_error(val_y,preds)]
    a_series = pd.Series(to_append, index = model_scores.columns)
    model_scores = model_scores.append(a_series, ignore_index=True)

    return preds, model_scores, xg_model


def run_gbr(train_X,train_y,val_X,val_y):
    """[summary]

    Args:
        train_X ([type]): [description]
        train_y ([type]): [description]
        val_X ([type]): [description]
        val_y ([type]): [description]

    Returns:
        [type]: [description]
    """    

    model_name = "Gradient Boosted Regression"
    run=1

    model_scores = pd.DataFrame(columns = ['model_name', 'run', 'run_notes', 'accuracy', 'RMSE', 'MAE'])

    gbr_model = GradientBoostingRegressor(n_estimators=3460, learning_rate=0.01,
                                    max_depth=2, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10,
                                    loss='huber', random_state=5).fit(train_X,train_y)
    gbr_preds = gbr_model.predict(val_X)

    to_append = [model_name, run, None, r2_score(val_y,gbr_preds), mean_squared_error(val_y,gbr_preds), 
                    mean_absolute_error(val_y,gbr_preds)]
    a_series = pd.Series(to_append, index = model_scores.columns)
    model_scores = model_scores.append(a_series, ignore_index=True)

    return gbr_preds, model_scores, gbr_model

def run_lgbm(train_X,train_y,val_X,val_y):
    """[summary]

    Args:
        train_X ([type]): [description]
        train_y ([type]): [description]
        val_X ([type]): [description]
        val_y ([type]): [description]

    Returns:
        [type]: [description]
    """    
    
    model_name = "LightGBM"
    run=1
    
    model_scores = pd.DataFrame(columns = ['model_name', 'run', 'run_notes', 'accuracy', 'RMSE', 'MAE'])

    lgbm_model = LGBMRegressor(objective='regression',num_leaves=6,
                            learning_rate=0.05, n_estimators=720,
                            max_bin=55, bagging_fraction=0.5, bagging_freq=5, 
                            feature_fraction=0.2319, feature_fraction_seed=9, 
                            bagging_seed=9, min_data_in_leaf=6, 
                            min_sum_hessian_in_leaf=8).fit(train_X, train_y)
    lgbm_preds = lgbm_model.predict(val_X)

    to_append = [model_name, run, None, r2_score(val_y,lgbm_preds), mean_squared_error(val_y,lgbm_preds), 
                    mean_absolute_error(val_y,lgbm_preds)]
    a_series = pd.Series(to_append, index = model_scores.columns)
    model_scores = model_scores.append(a_series, ignore_index=True)

    return lgbm_preds, model_scores, lgbm_model

def run_ridgecv(n_splits,train_X,train_y,val_X,val_y):
    """[summary]

    Args:
        n_splits ([type]): [description]
        train_X ([type]): [description]
        train_y ([type]): [description]
        val_X ([type]): [description]
        val_y ([type]): [description]

    Returns:
        [type]: [description]
    """    
    
    model_name = "ridge Regressor"
    model_scores = pd.DataFrame(columns = ['model_name', 'run', 'run_notes', 'accuracy', 'RMSE', 'MAE'])
    run=1
    
    kfolds = KFold(n_splits, shuffle=False)
    
    r_alphas = [.0001, .0003, .0005, .0007, .0009, 
          .01, 0.05, 0.1, 0.3, 1]
    
    ridge_model = make_pipeline(RobustScaler(), RidgeCV(alphas = r_alphas,
                                    cv=kfolds)).fit(train_X, train_y)

    ridge_preds = ridge_model.predict(val_X)

    to_append = [model_name, run, None, r2_score(val_y,ridge_preds), mean_squared_error(val_y,ridge_preds), 
                    mean_absolute_error(val_y,ridge_preds)]
    a_series = pd.Series(to_append, index = model_scores.columns)
    model_scores = model_scores.append(a_series, ignore_index=True)

    return ridge_preds, model_scores, ridge_model

def run_lassocv(n_splits, alphas, train_X,train_y,val_X,val_y):
    """Runs the LassoCV regressor on a given dataset

    Args:
        n_splits ([int]): [Number of splits input to kfolds]
        alphas ([list]): [Parameter for the model]
        train_X ([DataFrame]): [Features in the training dataset]
        train_y ([DataFrame]): [Targets in the training dataset]
        val_X ([DataFrame]): [Features in the validation dataset]
        val_y ([DataFrame]): [Targets in the validation set]

    Returns:
        [list]: [A list containing the lasso_preds DataFrame, the model_scores DataFrame, and the trained model]
    """    
    
    model_name = 'LassoCV'
    model_scores = pd.DataFrame(columns = ['model_name', 'run', 'run_notes', 'accuracy', 'RMSE', 'MAE'])
    run=1

    kfolds = KFold(n_splits, shuffle=False)
    
    lasso_model = make_pipeline(RobustScaler(),
                             LassoCV(max_iter=1e7,
                                    alphas = alphas,
                                    random_state = 42, cv = kfolds)).fit(train_X, train_y)

    lasso_preds = lasso_model.predict(val_X)

    to_append = [model_name, run, None, r2_score(val_y,lasso_preds), mean_squared_error(val_y,lasso_preds), 
                    mean_absolute_error(val_y,lasso_preds)]
    a_series = pd.Series(to_append, index = model_scores.columns)
    model_scores = model_scores.append(a_series, ignore_index=True)

    return lasso_preds, model_scores, lasso_model

def run_elasticnetcv(n_splits, e_l1ratio, e_alphas, train_X,train_y,val_X,val_y):
    """[summary]

    Args:
        n_splits ([type]): [description]
        e_l1ratio ([type]): [description]
        e_alphas ([type]): [description]
        train_X ([type]): [description]
        train_y ([type]): [description]
        val_X ([type]): [description]
        val_y ([type]): [description]

    Returns:
        [type]: [description]
    """   
     
    model_name = 'ElasticNetCV'
    model_scores = pd.DataFrame(columns = ['model_name', 'run', 'run_notes', 'accuracy', 'RMSE', 'MAE'])
    run=1
    
    kfolds = KFold(n_splits, shuffle=False)
    
    elastic_model= make_pipeline(RobustScaler(), 
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas, 
                                        cv=kfolds, l1_ratio=e_l1ratio)).fit(train_X, train_y)

    elastic_preds = elastic_model.predict(val_X)
    
    to_append = [model_name, run, None, r2_score(val_y,elastic_preds), mean_squared_error(val_y,elastic_preds), 
                    mean_absolute_error(val_y,elastic_preds)]
    a_series = pd.Series(to_append, index = model_scores.columns)
    model_scores = model_scores.append(a_series, ignore_index=True)

    return elastic_preds, model_scores, elastic_model

def run_stackedregressor(n_splits, r_alphas, alphas2, e_alphas, e_l1ratio, train_X, train_y, val_X, val_y):
    
    model1_name = 'StackedCVRegressor'
    
    model_scores = pd.DataFrame(columns = ['model_name', 'run', 'run_notes', 'accuracy', 'RMSE', 'MAE'])
    
    run=1
    
    kfolds = KFold(n_splits, shuffle=False)
    
    #setup model layers
    ridge = make_pipeline(RobustScaler(), RidgeCV(alphas = r_alphas,
                                    cv=kfolds)).fit(train_X, train_y)

    lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas = alphas2,
                              random_state = 42)).fit(train_X,train_y)

    elasticnet = make_pipeline(RobustScaler(), 
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas, 
                                         l1_ratio=e_l1ratio)).fit(train_X,train_y)

    lgbm_model = LGBMRegressor(objective='regression',num_leaves=5,
                                      learning_rate=0.05, n_estimators=720,
                                      max_bin = 55, bagging_fraction = 0.8,
                                      bagging_freq = 5, feature_fraction = 0.2319,
                                      feature_fraction_seed=9, bagging_seed=9,
                                      min_data_in_leaf =6, 
                                      min_sum_hessian_in_leaf = 11).fit(train_X,train_y)
    
    gbr_model = GradientBoostingRegressor().fit(train_X,train_y)

    include_models = (ridge,lasso,elasticnet,gbr_model,lgbm_model)
    meta_regressor = gbr_model

    #stack
    stack_gen = StackingCVRegressor(regressors=include_models, 
                               meta_regressor=meta_regressor,
                               use_features_in_secondary=True)

    #prepare dataframes
    stackX = np.array(train_X)
    stacky = np.array(train_y)
    stack_gen_model = stack_gen.fit(stackX, stacky)
    
    em_preds = elasticnet.predict(val_X)
    lasso_preds = lasso.predict(val_X)
    ridge_preds = ridge.predict(val_X)
    stack_gen_preds = stack_gen_model.predict(val_X)
    lgbm_preds = lgbm_model.predict(val_X)
    gbr_preds = gbr_model.predict(val_X)
        
    to_append1 = [model1_name, run, None, r2_score(val_y,stack_gen_preds), mean_squared_error(val_y,stack_gen_preds), 
                    mean_absolute_error(val_y,stack_gen_preds)]
    a_series1 = pd.Series(to_append1, index = model_scores.columns)
    model_scores1 = model_scores.append(a_series1, ignore_index=True)
    
    return stack_gen_preds, model_scores1, stack_gen_model, em_preds, lasso_preds, ridge_preds, stack_gen_preds, lgbm_preds, gbr_preds
    
def run_stack_weighted_regressor(run, weights, em_preds, lasso_preds, ridge_preds, stack_gen_preds, lgbm_preds, gbr_preds, val_X, val_y):
    model_name = "Stacked Weighted Regressor"
    model_scores = pd.DataFrame(columns = ['model_name', 'run', 'run_notes', 'accuracy', 'RMSE', 'MAE'])
              
    w1, w2, w3, w4, w5 = weights
        
    stack_preds_run = ((w1*em_preds) + (w2*lasso_preds) + (w3 * gbr_preds ) + (w4*lgbm_preds) + (w5*stack_gen_preds))
    
    to_append = [model_name, run, None, r2_score(val_y,stack_preds_run), mean_squared_error(val_y,stack_preds_run), 
                    mean_absolute_error(val_y,stack_preds_run)]
    a_series = pd.Series(to_append, index = model_scores.columns)
    model_scores = model_scores.append(a_series, ignore_index=True)

    return stack_preds_run, model_scores