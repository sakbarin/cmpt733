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
        """
            Write your code!
        """

    def verification(self, cand_df, threshold):
        """
            Write your code!
        """

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
