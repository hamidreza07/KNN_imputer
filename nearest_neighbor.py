
# Test distance function
# dataset = (df.select_dtypes(include=np.number).values).tolist()


#%%
from random import seed
import pandas as pd
from math import sqrt
import warnings
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
from statistics import *
from random import randrange
from sklearn.model_selection import KFold
#%%

#https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

class n_neighbors:
    def __init__(self,return_distance=True,normalization=False):
        
        self.return_distance=return_distance
        self.normalization=normalization
        warnings.filterwarnings('ignore')

    def __euclidean_distance(self,row1, row2):
        distance = 0.0
        for i in range(len(row1)-1):
            if type(row1[i])==str:
                if row1[i]==row2[i]:
                    distance+=0
                else:
                    distance+=1
            else:
                distance += (row1[i] - row2[i])**2
        return sqrt(distance),row2[-1]
    
    def __predict_classification(self,train, test_row, num_neighbors):
        neighbors = self.__get_neighbors_test(train, test_row, num_neighbors)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction

    # Locate the most similar neighbors
    def __convert_list(self,df):
        val=[]
        for item in df.values:
            val.append(list(map(lambda x: x.replace('nan', ''), [str(i) for i in item])))
        return val
    
    def __k_nearest_neighbors(self,train, test, num_neighbors):
        predictions = list()
        for row in test:
            output =self.__predict_classification(train, row, num_neighbors)
            predictions.append(output)
        return(predictions)
            

    
    def __evaluate_algorithm(self,dataset, algorithm, *args):
        folds = self.__cross_validation_split(dataset, 10)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.__accuracy_metric(actual, predicted)
            scores.append(accuracy[0])
        return scores,accuracy[1]

    def __accuracy_metric(self,actual, predicted):
        error = 0
        count=0
        if actual[0].isdigit() or actual[0].replace('.','',1).isdigit() and actual[0].count('.') < 2:
            cat=False
            for i in range(len(actual)):
                
                error +=(eval(actual[i]) - eval(predicted[i]))**2
                count += 1
            return error / count ,cat
    
        else :
            cat=True
            correct = 0
            for i in range(len(actual)):
                if actual[i]==predicted[i]:
                    correct+=1
            return  correct / float(len(actual)) * 100.0,cat  

    
        # Split a dataset into k folds
    def __cross_validation_split(self,dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for _ in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
       
    
        return dataset_split

    
    def __get_neighbors_test(self,train, test_row, num_neighbors):
        distances = list()
        for train_row in train:
            distances.append((train_row, self.__euclidean_distance(test_row, train_row)[0]))
        distances.sort(key=lambda tup: tup[1])
        return [distances[i][0] for i in range(num_neighbors)]



    def get_neighbors(self,df, test_row):
        if self.normalization:
            df[df.select_dtypes(np.number).columns]=pd.DataFrame(MinMaxScaler().fit_transform(X=df[df.select_dtypes(np.number).columns]),columns=df.select_dtypes(np.number).columns)
        val=df.values.tolist()
        for i in range(len(df)):
            val[i].append([df.index[i]])
        distances=[self.__euclidean_distance(test_row, row) for row in val]
        distances.sort(key=lambda tup: tup[0])

        
        if self.return_distance:
            return [distances[i][0] for i in range(1,self.accuracy(df)[0]+1)],[distances[i][1][0] for i in range(1,self.accuracy(df)[0]+1)]
        else:
            return [distances[i][1][0] for i in range(1,self.accuracy(df)[0]+1)]
        

    
    
        
        
        
    def accuracy(self,df):
        val=self.__convert_list(df.dropna().sample(50))
        
        seed(1)
        num_neighbors = [i for i in range(3,5)]
        best_k=0
        max_a=0
        max_accuracy=np.inf
        
        for i in num_neighbors:
            accuracy=sum(self.__evaluate_algorithm(val, self.__k_nearest_neighbors, i)[0])/float(len(self.__evaluate_algorithm(val, self.__k_nearest_neighbors, i)[0]))
            if self.__evaluate_algorithm(val, self.__k_nearest_neighbors, i)[1]:
                if accuracy>max_a:
                    max_a =accuracy
                    best_k=i
            else:
                
                if accuracy<max_accuracy:
                    max_accuracy =accuracy
                    best_k=i
        if self.__evaluate_algorithm(val, self.__k_nearest_neighbors, i)[1]:
            return best_k,max_a
        else:
            return best_k,max_accuracy
            
        

def test1():
        df=pd.read_csv(r'1.csv')
        print(df)
        

        warnings.filterwarnings('ignore')
        imp=n_neighbors().accuracy(df)
        print(imp)
        # df_scale=df.copy()
        
        # df_scale[df_scale.select_dtypes(np.number).columns]=pd.DataFrame(MinMaxScaler().fit_transform(X=df_scale[df_scale.select_dtypes(np.number).columns]),columns=df_scale.select_dtypes(np.number).columns)
        # df_null=df_scale[pd.isnull(df_scale["0"])].fillna((df_scale.mean())).drop("0",1).apply(lambda x: x.fillna(x.value_counts().index[0]))
        # null_val=df_null.values
        # df_not_null=df[~pd.isnull(df["0"])].fillna((df.mean())).apply(lambda x: x.fillna(x.value_counts().index[0]))
        # df_not_null_scale=df_scale[~pd.isnull(df_scale["0"])].fillna((df_scale.mean())).apply(lambda x: x.fillna(x.value_counts().index[0]))
    
        # df_not_null_scale=df_not_null_scale.reset_index(drop=True)
        # # print(df_not_null)
        # for count in range(len(null_val)):
        #     indices = n_neighbors(return_distance=False).get_neighbors(df_not_null_scale,null_val[count])
        #     print(indices)
        #     print(df_not_null.iloc[2]["0"])
        #     # print(df.iloc[df_null.index[count]])  
        #     # print(mean([df_not_null.iloc[j][0] for j in indices])) 
        #     df.at[df_null.index[count],"0"]=mean([df_not_null.iloc[j]["0"] for j in indices])
        #     print(df.iloc[df_null.index[count]])   
   
        #     break
   
        


if __name__=='__main__':
    test1() 