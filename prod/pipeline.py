#!/usr/bin/python
# -*- coding: utf-8 -*-

import psql_handler
import numpy as np
import re
import pandas as pd 
import seaborn as sns
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib


def _data_engineering():
    """
    Function to simulate the data engineering section: 
        Collection, cleansing and formatting and storage of data.
    """
    data = pd.read_csv('titanic_dataset.csv')
    psql_handler.write_data(data, 'titanic', 'PassengerId','5432', 'pgdb')
    print("Data uploaded")


def _distance_category(desc, arr1):
    '''
    e.g.

    arr = np.array([0,0,3])
    s = distance_category(desc, arr)
    res = 2
    '''
    scores = {}
    for group in [0,1,2,3,4,5,6,7]:
        scores[distance.euclidean(desc[group], arr1)] = group
    return scores[min(scores.keys())]


def _create_vectors(data):
    ''' 
    The results should be a dict with bins as keys and 3 value vectors as values.
    e.g.:
        {
        1:[2.5,4.3,1],
        2:[3.2, 2.1,2],
        .
        . etc..
        }
    '''
    desc = {}
    for group in [0,1,2,3,4,5,6,7]:
        desc[group] = []
        desc[group].append(round(data[data['age_group']==group]['SibSp'].mean(),2))
        desc[group].append(round(data[data['age_group']==group]['Parch'].mean(), 2))
        desc[group].append(round(data[data['age_group']==group]['Pclass'].mean(), 2))
        desc[group] = np.array(desc[group])
    return desc

def _estimate_age(data):
    """ 
    e.g. bucket:age estimation 
        {0: 4, 1: 15, 2: 21, 3: 29, 4: 38, 5: 48, 6: 59, 7: 68}
    """
    age_estim = {}
    for group in [0,1,2,3,4,5,6,7]:
        age_estim[group] = int(data[data['age_group']==group]['Age'].mean())
    return age_estim


def _find_title(name):
    # Some titles I watched on a quick search
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major',
                'Dr', 'Ms', 'Capt','Don', 'other']
    patterns = {
        re.compile(r'Mrs', re.IGNORECASE): 'Mrs',
        re.compile(r'Mr', re.IGNORECASE): 'Mr',
        re.compile(r'Master', re.IGNORECASE): 'Master',
        re.compile(r'Miss', re.IGNORECASE): 'Miss',
        re.compile(r'Major', re.IGNORECASE): 'Major',
        re.compile(r'Dr', re.IGNORECASE): 'Dr',
        re.compile(r'Ms', re.IGNORECASE): 'Ms',
        re.compile(r'Capt', re.IGNORECASE): 'Capt',
        re.compile(r'Don', re.IGNORECASE): 'Don',
    }
    title = 'Other'
    for patt, val in patterns.items():
        if re.search(patt, name):
            title = val
    return title


class MLpipeline:
    """ 
    Machine learning pipeline challenge
    """
    def data_processing(self, data):
        print("Start processing data")
        # Filling missing values
        print("Filling missing values")
        data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode().iloc[0])
        print("Embarked Filled")
        # if fare missing fill with mean
        data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

        #  AGE filling missing values
        ''' 
        Buckets:
            0: 0-11, 1:12-17, 2:18-24, 3:25-34, 4:35-44, 5:45-54, 6:55-64, 7:65-80, 8:NaN
        '''
        bins = (12,18,25,35,45,55,65,80)
        categories = np.digitize(data['Age'], bins)
        data['age_group'] = categories
        desc = _create_vectors(data=data)
        # find null values on Age
        data['isnull'] = data['Age'].isnull()
        data['age_group'] = data.apply(lambda x: 
                                       _distance_category(desc, np.array([x['SibSp'], x['Parch'], x['Pclass']])) 
                                       if x['isnull'] else x['age_group'], axis=1)
        print("Age group Added")
        age_estim = _estimate_age(data=data)
        data['Age'] = data.apply(lambda x: age_estim[x['age_group']] if x['isnull'] else x['Age'], axis=1)
        data = data.drop('isnull', axis=1)
        print("Age Filled")

        # Family size
        data['fam_size'] = data.apply(lambda x: x['SibSp']+x['Parch'], axis=1)
        print("Fam size feature Added")

        # titles
        data['title'] = data['Name'].map(lambda x: _find_title(x))
        print("title feature Added")        

        # Vector Encoding
        print("Start Vector Encoding...")
        data = pd.concat([data, pd.get_dummies(data['Embarked'],prefix='embarked')],axis=1)
        data = data.drop('Embarked', axis=1)

        data['sex_bi'] = data['Sex'].map(lambda x: 1 if x=='male' else 0)
        data = data.drop('Sex', axis=1)

        label_encoder = preprocessing.LabelEncoder()
        data['title_label'] = label_encoder.fit_transform(data['title'])
        data = data.drop('title', axis=1)
        print("Vector Encoding DONE")

        # dropping some features
        print("Dropping useless features")
        # useless
        data = data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)
        # after first model and check corr
        data = data.drop(['Age', 'fam_size'], axis=1)
        # after 2nd iteration and check corr between S and C
        data = data.drop(['embarked_S'], axis=1)
        # after 3rd iteration and check feature importance
        data = data.drop(['embarked_C', 'embarked_Q'], axis=1)

        print("Pre-processing DONE")

        return data

    def training(self, data):
        print("Training RF Model")
        X_train = data.drop("Survived", axis=1)
        Y_train = data["Survived"]

        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                               max_depth=16, max_features=0.7, max_leaf_nodes=39,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=922,
                               n_jobs=-1, oob_score=True, random_state=42, verbose=0,
                               warm_start=False)
        rf.fit(X_train, Y_train)

        # save the model to disk
        filename = 'rf_ppl.sav'
        joblib.dump(rf, filename)
        
        print("Model Saved to disk")


    def predict(self, dataset):
        print("Loading Classifier")
        # load the model from disk
        loaded_model = joblib.load('rf_ppl.sav')
        result = loaded_model.predict(dataset)
        print("Predictions DONE")
        return result

    def load_resuts(self, results, port):
        # data = pd.read_csv('results_dataset.csv')
        psql_handler.write_data(results, 'predictions', 'PassengerId', port, 'results')
        print("Results uploaded")

def main():
    # simulate data eng process
    _data_engineering()
    
    # initialize pipeline
    ppl = MLpipeline()

    # DownLoad Training Data
    query = "SELECT * FROM titanic;"
    data = psql_handler.read_data(query, '5432', 'pgdb')
    print("Training Data LOADED")
    # processing
    df = ppl.data_processing(data)
    # training
    ppl.training(data=df)
    
    # read external data from customers or some other DB
    external_df = pd.read_csv('external_dataset.csv')
    # processing
    df = ppl.data_processing(external_df)
    # predict
    res = ppl.predict(df)
    df['PassengerId'] = external_df['PassengerId']
    df['Survived'] = res
    # load results
    ppl.load_resuts(df, '5433')
    print("Pipeline DONE")


if __name__ == "__main__":
    main()
