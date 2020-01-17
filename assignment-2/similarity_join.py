# similarity_join.py
import re
import pandas as pd

class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)

    def concat_columns(self, cols):
        # concat columns values with a space between
        return ' '.join(cols.values.astype(str))
    
    def tokenize(self, col_value):
        # tokenize input column value
        values = re.split(r'\W+', col_value.strip())
        return [value.lower() for value in values if (value.lower() != '' and value.lower() != 'nan')]
    
    def preprocess_df(self, df, cols):         
        # concat provided columns in new column called concat_col        
        df['concat_col'] = df[cols].apply(self.concat_columns, axis=1)
        
        # tokenize and lower concatenated columns and save it in joinKey column
        df['joinKey'] = df['concat_col'].apply(self.tokenize)
        
        # drop previously concatenated column
        df.drop('concat_col', inplace=True, axis=1)

        # return result dataframe
        return df

    def filtering(self, df1, df2):
        # define a new column to explode joinKey
        df1['explodedJoinKey'] = df1['joinKey']
        df2['explodedJoinKey'] = df2['joinKey']
        
        # explode joinKey column
        df1_exploded = df1[['id', 'joinKey', 'explodedJoinKey']].explode('explodedJoinKey')
        df2_exploded = df2[['id', 'joinKey', 'explodedJoinKey']].explode('explodedJoinKey')
        
        # join two dataframes on exploded joinKey to find rows with shared elements
        df_joined = pd.merge(df1_exploded, df2_exploded, on='explodedJoinKey', how='inner')
        
        # drop exploded joinKey
        df_dropped = df_joined.drop(['explodedJoinKey'], axis=1)
        
        # rename columns after join
        df_renamed = df_dropped.rename(columns={'id_x': 'id1', 'joinKey_x': 'joinKey1', 'id_y': 'id2', 'joinKey_y': 'joinKey2'})
        
        # drop duplicate rows of (id1, id2)
        df_filtered = df_renamed.drop_duplicates(['id1', 'id2'], keep='first')
        
        # return result dataframe
        return df_filtered

    def intersect(self, cols):
        # this function is used to calcuate jaccard value for each record
        joinKey1 = set(cols[0])
        joinKey2 = set(cols[1])
        
        intersection_count = len(joinKey1.intersection(joinKey2))
        union_count = len(joinKey1.union(joinKey2))
        
        return (intersection_count / union_count)
        
    def verification(self, cand_df, threshold):
        # computer jaccard value using self.intersect function
        cand_df['jaccard'] = cand_df[['joinKey1', 'joinKey2']].apply(self.intersect, axis=1)
        
        # select records with jaccard value greater than threshold
        result_df = cand_df.loc[cand_df['jaccard'] >= threshold]
        
        # return results
        return result_df

    def evaluate(self, result, ground_truth):
        """
            Write your code!
        """

    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0])) 

        cand_df = self.filtering(new_df1, new_df2)
        print ("After Filtering: %d pairs left" %(cand_df.shape[0]))

        result_df = self.verification(cand_df, threshold)
        print ("After Verification: %d similar pairs" %(result_df.shape[0]))

        return result_df



if __name__ == "__main__":
    er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

    result = result_df[['id1', 'id2']].values.tolist()
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))
