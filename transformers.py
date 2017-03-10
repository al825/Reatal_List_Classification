# define the transformers
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.base import TransformerMixin
from bs4 import BeautifulSoup 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import random
from collections import defaultdict
import re
import time 


class VariableExtractor(TransformerMixin):
    '''Extract variable(s).'''    
    def __init__(self, variables):
        self.variables = variables
        
    def fit(self, *_):
        return self
    
    def transform(self, dataset):
        return dataset[self.variables]
        
class DimOneUp(TransformerMixin):
    '''Turn Series into array with 2 dimensions'''
    
    def fit(self, *_):
        return self
    
    def transform(self, series):
        return series.reshape((series.shape[0], 1))
        
class RatioCreator(TransformerMixin):
    '''Create new variable as the ratio of two variables.'''
    def __init__(self, variable1, variable2):
        self.variable1 = variable1
        self.variable2 = variable2
        
    def fit(self, *_):
        return self
    
    def transform(self, dataset):
        return dataset[self.variable1]/dataset[self.variable2].apply(lambda x: x if x != 0 else 1)

class RatioCreator(TransformerMixin):
    '''Create new variable as the ratio of two variables.'''
    def __init__(self, variable1, variable2):
        self.variable1 = variable1
        self.variable2 = variable2
        
    def fit(self, *_):
        return self
    
    def transform(self, dataset):
        return dataset[self.variable1]/dataset[self.variable2].apply(lambda x: x if x != 0 else 1)
        
        
# cluster longitutide and latitute
class LLCluster(TransformerMixin):
    '''Cluster longitude and latitude.'''
    def __init__(self, n_clusters, **kargs):
        self.model = MiniBatchKMeans(n_clusters=n_clusters, **kargs)
        
    def fit(self, dataset, *_):
        self.model.fit(dataset)
        return self
    
    def transform(self, dataset):
        return self.model.predict(dataset)
        
        
class VariableLength(TransformerMixin):
    '''Get the length of the variable when it is a list.'''
    
    def fit(self, *_):
        return self
    
    def transform(self, dataseries):
        return dataseries.apply(len)
        
     
class FeatureCleanser(TransformerMixin):
    '''Clean the features
       Typical features in the data set: ['featureA', 'featureB']
       But some features are like ['featureA**featureB'].
       Turn those features into ['featureA', 'featureB']
    '''
    
    def __init__(self, spliter=['*', '.', '^']):
        self.spliter = spliter
    
    def fit(self, *_):
        return self
    
    def transform(self, dataset):
        return dataset['features'].apply(self.feature_clean)
            
    def feature_clean(self, feature_list):
        '''Clean the features.'''
        for ff in feature_list: 
            if any(x in ff for x in self.spliter):
                feature_list.remove(ff)
                ff = re.sub('[{}]+'.format('|'.join(self.spliter)), ',', ff)
                #ff = re.sub('[*|.|^]+', ',', ff)
                # remove the ',' at the beginning and at the end of the string
                ff = re.sub('^[,]|[,]$', '', ff)
                feature_list += ff.split(',')
        # clean the text, strip and lower case
        return [f.strip().lower() for f in feature_list]
        
class DiffFeatCounts(TransformerMixin):
    '''For the Feature record, create a data set to count the most different features across classes.
    '''
    def __init__(self, sample_size=1000, min_freq=200, n_iter=10, threshold=0.5, random_state=0):
        self.sample_size = sample_size
        self.min_freq = min_freq
        self.n_iter = n_iter
        self.threshold = threshold
        self.random_state = random_state
    
    def fit(self, dataseries, y, *_):
        self.fit_set = dataseries
        self.y = y
        self.diff_feat = self.find_DiffFeat()
        return self
    
    def transform(self, dataseries):
        return dataseries.apply(self.feature_counts)

    def feature_counts(self, features):
        '''For each 'features' record, count the frequency of the different features in the feature record and
           create the data frame based on the counts. 
        '''
        feat_series = pd.Series([0]*len(self.diff_feat), index=self.diff_feat)
        for f in self.diff_feat:
            feat_series[f] = features.count(f)
        return feat_series
    
    def find_DiffFeat(self):
        '''Find the most different features across the interest levels.
           Criteria: features appear > min_freq
                     any(%interest_level > threshold)
           Return a list of different features.                      
        '''
        random.seed(self.random_state)
        feature_df = defaultdict(lambda: defaultdict(int))
        # feature_df = defaultdict(defaultdict(int)), this did not work
        # feature_df = {'featureA': {'low':30, 'medium':10, 'high':2}, 'featureB': ...}
        data_addy = pd.concat([self.fit_set, self.y], axis=1)
        # Iterate the process. In each iteration, sample a subset with equal number of each interest level
        for n in range(self.n_iter):
            data_temp = pd.DataFrame(columns=['features', 'interest_level'])
            # for each interest level, sample equal size 
            for i in self.y.unique():
                data_temp = data_temp.append(data_addy[self.y==i].sample(n=self.sample_size))
            for ind in data_temp.index:
                for f in data_temp.loc[ind, 'features']:
                    feature_df[f][data_temp.loc[ind, 'interest_level']] += 1         
        diff_feat = [fk for fk, fv in feature_df.items() if sum(fv.values()) >= self.min_freq 
                     and max(fv.values())/sum(fv.values()) >= self.threshold
                    ]
        return diff_feat
      
      
class DescriptionWordCounts(TransformerMixin):
    '''Count the number of words innthe description.'''
    def __init__(self, tokenizer=RegexpTokenizer(r'\w+')):
        self.tokenizer = tokenizer
        
    def fit(self, *args):
        return self
    
    def transform(self, dataset):
        return dataset['description'].apply(lambda x: len(self.tokenizer.tokenize(x)))
        
  
class DescriptionProcessor(TransformerMixin):
    '''Process the description.'''
    def __init__(self, stemmer=SnowballStemmer('english'), tokenizer=RegexpTokenizer(r'\w+'), min_df=5000, stop_words='english', *args):
        self.vectorizer = TfidfVectorizer(min_df=min_df, stop_words=stop_words, preprocessor=lambda p: self.preprocessor(p, stemmer=stemmer, tokenizer=tokenizer))
        
    def fit(self, dataset, *_):
        self.vectorizer.fit(dataset['description'])
        return self
    
    def transform(self, dataset):
        return self.vectorizer.transform(dataset['description']).toarray()

    def preprocessor(self, text, stemmer, tokenizer):
        '''Preprocess the description.'''
        # remove numbers
        text = re.sub('[0-9]*', '', text)
        #tokenize the description, stem each word and link words back into sentences
        text = ' '.join([stemmer.stem(x) for x in tokenizer.tokenize(text)])
        return text
        
class CatVariableCounts(TransformerMixin):
    '''Count number of lists for each category of the categorical variable.
       e.g. How many lists does a manager have    
    '''       
    
    def fit(self, dataseries, *_):
        self.catcounts = dataseries.value_counts()
        return self
        
    def transform(self, dataseries):
        return dataseries.apply(lambda x: self.catcounts[x] if x in self.catcounts.index else 0)
        
     
class CatVariableIndicator(TransformerMixin):
    '''Find the category levels of the categorical variable which have more low or more medium or more high. 
       Criteria: Frequency of the category > min_list
                 For a category, the percent of any interest_level greater than the corresponding threshold. 
    '''
    def __init__(self, min_list=4, threshold={'low': 0.8, 'medium': 0.6, 'low': 0.4}):
        '''
            Args:
                variable: name of the categorical variable
                min_list: minimal number of the list the category should have
                threshold: a dictionary haing the thresholds for each interest level (thresholds in percentage)
        
        '''
        self.min_list = min_list
        self.threshold = threshold
        self.hml_features = defaultdict(set)
        
    def fit(self, dataseries, y, *_):
        cat_counts = dataseries.value_counts()
        self.ylevels = y.unique()
        # restrict to records with listings more than the min_list
        elig_data = dataseries[dataseries.isin(cat_counts[cat_counts>=self.min_list].index.values)]
        elig_y = y[dataseries.isin(cat_counts[cat_counts>=self.min_list].index.values)]
        for category in elig_data.unique():
            y_pectages = self.y_pect(elig_data, elig_y, category)
            for ylevel in self.ylevels: 
                try:
                    if y_pectages[ylevel] >= self.threshold[ylevel]:
                        self.hml_features[ylevel].add(category)
                        break
                except:
                    pass
        return self
    
    def transform(self, dataseries):
        return dataseries.apply(self.single_transform)
        
    def y_pect(self, dataseries, y, category):
        return y[dataseries==category].value_counts(normalize=True)
    
    def single_transform(self, category):
        return pd.Series([(category in v) for v in self.hml_features.values()])                

class DateProcessor(TransformerMixin):
    '''Returns the year, month and hour of the date'''
    def __init__(self, wantyear=False, wantmonth=False, wanthour=True):
        self.wantyear= wantyear
        self.wantmonth = wantmonth
        self.wanthour = wanthour
        
    def fit(self, *_):
        return self
    
    def transform(self, dataset):
        return dataset['created'].apply(self).iloc[:, [self.wantyear, self.wantmonth, self.wanthour]]
    
    def process_date(self, date):
        year = date[:4]
        month = date[5:7]
        hour = date[11:13]
        return pd.Series([year, month, hour], index=('year', 'month', 'hour'))

        
class AddressCleanser(TransformerMixin):
    '''Clean the address.
        Strip and lowcase the address. Standardize synonyms into one expression. 
    '''
    def __init__(self, synonyms=[(r'([\d])((st)|(nd)|(rd)|(th))', r'\1'),(r'( street)|( st)', r' st.'), 
                                 (r'( avenue)|( ave)', r' ave.'), (r'(w )', r'west '),(r'(e )', r'east '), 
                                 (r'(n )', r'north '), (r'(s )', r'south ')], variable='display_address'):
        self.synonyms = synonyms
        self.variable = variable
        
    def fit(self, *_):
        return self
    
    def transform(self, dataset):
        return dataset[self.variable].apply(self.clean_address)
        
    def clean_address(self, address):
        address = address.strip()
        address = address.lower()        
        for s1, s2 in self.synonyms:
            address = re.sub(s1, s2, address)
        return address


