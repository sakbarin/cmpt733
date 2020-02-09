# anomaly_detection.py
import pandas as pd
import numpy as np
import ast 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class AnomalyDetection():

    def scaleNum(self, df, indices):    
        df_normal = pd.DataFrame(df['features'].tolist())
        df_normal = df_normal.reset_index()
        df_normal.rename(columns={'index': 'id'}, inplace=True)
        
        # add 1 to indices [because of index column]
        indices = np.add(indices, 1)

        # iterate through all indices
        for index in indices:
            
            # find mean of column index
            mean = df_normal.iloc[:,index].mean()

            # find x - mean
            df_normal['x_mean'] = df_normal.iloc[:,index] - mean

            # find (x - mean) ^ 2
            df_normal['(x_mean)pow2'] = df_normal['x_mean'] * df_normal['x_mean']

            # find variance
            variance = df_normal['(x_mean)pow2'].sum() / (df_normal['(x_mean)pow2'].count() - 1)
            
            # find std dev
            standard_deviation = np.sqrt(variance)

            # set column value to (x - mean) / std dev
            df_normal.iloc[:,index] = df_normal['x_mean'] / standard_deviation
            
            # drop temp columns
            df_normal.drop(columns=['x_mean', '(x_mean)pow2'], inplace=True)

        # replace all nan values to 0 [nan because of div by zero]
        df_normal.fillna(0, inplace=True)
        
        # merge all values to feature column
        df_normal['features'] = df_normal.iloc[:,1:].to_numpy().tolist()

        # return result
        return df_normal[['id', 'features']]


    
    def cat2Num(self, df, indices): 
        df_normal = df.reset_index()
        df_normal.rename(columns={'index': 'id'}, inplace=True)
        
        # keep column names
        columns = df_normal.columns

        # no. of columns added
        new_added_columns = 0
        
        # add 1 to indices [because of index column]
        indices = np.add(indices, 1)
        
        # iterate indicdes times
        for index in indices:            
            # get all distinct values in column index
            df_items = df_normal.iloc[:,index].drop_duplicates().reset_index(drop=True)
            
            # generate an identity matrix (with len(items) rows) for that column
            df_identity = pd.DataFrame(data=np.identity(len(df_items), dtype=int))
            
            # len(items) is number of new columns added
            new_added_columns = new_added_columns + len(df_items)

            # rename identity columns to corresponding value (0 -> http, 1 -> ftp, 0)
            for sub_index, item in df_identity.iterrows():
                df_identity.rename(columns={sub_index: df_items[sub_index]}, inplace=True)

            # concat columns of items and identity matrix 
            df_concat = pd.concat([df_items, df_identity], axis=1)

            # join main input matrix with concatenated matrix to have one hot encoding as columns of that matrix
            df_normal = pd.merge(left=df_normal, right=df_concat, how='inner', on=columns[index])
    
    
        # drop input columns [because we have one-hot encoding columns instead]
        df_normal.drop(columns=df_normal.columns[indices], inplace=True)
        
        # list of all new columns
        final_columns = list(df_normal.columns)
        
        # put columns in order [id, all generated one-hot encoding columns, rest of columns]
        df_normal = df_normal[final_columns[0:1] + final_columns[-1 * new_added_columns:] + final_columns[1:len(final_columns)-new_added_columns]]
    
        # create a feature column
        df_normal['features'] = df_normal.iloc[:,1:].to_numpy().tolist()
        
        return df_normal[['id', 'features']].sort_values('id')

    def detect(self, df, k, t):
        # init KMeans 
        kmeans = KMeans(n_clusters=k)
        
        # convert features to lists
        x_train = list(df['features'])      
    
        # fit training set
        kmeans.fit(x_train)

        # copy predicted labels to dataframe
        df_temp = df.copy()
        df_temp['cluster'] = kmeans.labels_

        # groupby clusters to get max, min, and each group count
        df_count = df_temp[['features', 'cluster']].groupby('cluster', as_index=False).count()
        df_count.rename(columns={'features': 'cluster_count'}, inplace=True)

        # get max cluster count
        n_max = df_count['cluster_count'].max()

        # get min cluster count
        n_min = df_count['cluster_count'].min()

        # merge temp dataframe and count of clusters to have a column showing count of that row's cluster        
        df_result = pd.merge(left=df_temp, right=df_count, on='cluster')

        # calculate score
        df_result['score'] = (n_max - df_result['cluster_count']) / (n_max - n_min)

        # return required columns and filter rows having value greater than threshold
        return df_result[['id', 'features', 'score']].loc[df_result['score'] >= t]
        

# set option to show features completely
pd.set_option('display.max_colwidth', -1)

# create object
ad = AnomalyDetection()

# dataset file name
file_name = 'logs-features-sample.csv'

# features column name
col_name = 'features'


# prepare dataframe
df = pd.read_csv(file_name, converters={col_name: ast.literal_eval})
df = pd.DataFrame(df[col_name].tolist())

# convert categorical to one-hot encoding
df1 = ad.cat2Num(df, [0,1])
print(df1)

# convert numerical to standardized numerical
df2 = ad.scaleNum(df1, list(range(12,48)))
print(df2)

# detect anomalies
df3 = ad.detect(df2, 8, 0.97)
print(df3)
