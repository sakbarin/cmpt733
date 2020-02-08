# anomaly_detection.py
import pandas as pd
import numpy as np

class AnomalyDetection():

    def normalizeDf(self, df):
        for i in range(len(df['features'][0])):
            df['f' + str(i + 1)] = df['features'].apply(lambda x: x[i])

        df.drop(columns=['features'], inplace=True)
        return df
    
    def scaleNum(self, df, indices):
        """
            Write your code!
        """


    def cat2Num(self, df, indices):
        # break feature list to some columns
        df_normal = self.normalizeDf(df)

        # add one to indices to (because of id column)
        indices = np.add(indices, 1)

        # keep column names
        columns = df_normal.columns

        # no. of columns added
        new_added_columns = 0
        
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
        
        # return [id, feature] dataframe
        return df_normal[['id', 'features']]

    def detect(self, df, k, t):
        """
            Write your code!
        """


#df = pd.read_csv('logs-features-sample.csv').set_index('id')
ad = AnomalyDetection()

data = [(0, ["http", "udt", 4]), \
        (1, ["http", "udf", 5]), \
        (2, ["http", "tcp", 5]), \
        (3, ["ftp", "icmp", 1]), \
        (4, ["http", "tcp", 4])]

df = pd.DataFrame(data=data, columns = ["id", "features"])

df1 = ad.cat2Num(df, [0,1])
print(df1)

#df2 = ad.scaleNum(df1, [6])
#print(df2)

#df3 = ad.detect(df2, 8, 0.97)
#print(df3)


