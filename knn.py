#%%
import pandas as pd
import numpy as np
# from generetor import *
from sklearn.neighbors import NearestNeighbors
from statistics import *
from time import process_time
import warnings
#%%vrb_level=1


class KNN_imputer:
    def __init__(self,trsh=.05,k=5,algorithm='auto',n_jobs=None,vrb_level=1):
        self.k=k
        self.verbose=vrb_level
        self.missing_treshold=trsh
        self.algorithm=algorithm
        self.n_jobs=n_jobs
        warnings.filterwarnings('ignore')
    
    def __type_detector(self,vals,A_dtypes):
        if not vals.empty:
                    categorical=[]
                    integer_col =[]
                    float_col =[]
                    boolian = []
                    for item in A_dtypes:
                        if A_dtypes[item].type==str  :
                            categorical.append(item)
                        elif  A_dtypes[item].type==np.bool_:
                            boolian.append(item)
                        elif  A_dtypes[item].type==np.intp:
                            integer_col.append(item)
                        elif  A_dtypes[item].type==np.float_:
                            float_col.append(item)
                   
                    return boolian,categorical,float_col,integer_col
    
    def __is_missing_acceptable(self,clmn_vals):
        miss_prc = clmn_vals.isnull().sum() / len(clmn_vals)

        if 0<=miss_prc<=self.missing_treshold:
            if self.verbose>0:
                print(f'the percent of missing in this column :{miss_prc}')
            return True

        elif miss_prc>self.missing_treshold:
            if self.verbose>0:
                print(f'the percent of missing in this column :{miss_prc}')
            return False
    
    def filler(self,df):
        # delete the column dou to threshold
        if len(df.index)>0:
            df.reset_index()
        df_new=pd.DataFrame()
        for clmn in  df.columns:
            if self.verbose>0:
                print("***************************************************")
                print("column name : ",clmn)
            cl_vals=df[clmn]
            miss=self.__is_missing_acceptable(cl_vals)
            if (miss)==False:
                df=df.drop(clmn,1)
                if self.verbose>0:
                    print(f'the {clmn} column would be deleted according to threshold')
            elif  (miss)==True:
                df_new[clmn]=df[clmn]
        if not df_new.empty:
            A_dtypes=dict(df_new.convert_dtypes().dtypes)
            boolian,categorical,float_col,integer_col=self.__type_detector(df_new,A_dtypes)
            if  integer_col:
                df_new=self.__int_filler(df_new,integer_col)
            if  float_col:
                df_new=self.__float_filler(df_new,float_col)
            if  categorical:
                df_new=self.__cat_filler(df_new,categorical)
            if  boolian:
                df_new=self.__bin_filler(df_new,boolian)
            df_new=self.__id_filler(df_new)
            return df_new      
        else:
            print('all dataframe had been deleted due to missing treshhold')
       
    def __cat_filler(self,df_new,vals):
        for item in vals: 
            df_null=df_new[ pd.isnull(df_new[item])].select_dtypes(np.number)
            if not df_null.empty:
                df_numeric=df_new[~ pd.isnull(df_new[item])].select_dtypes(np.number)
                indices = NearestNeighbors(n_neighbors=self.k, algorithm=self.algorithm,n_jobs=self.n_jobs)\
                    .fit(df_numeric).kneighbors(df_null.values,return_distance=False)
                for i in range(len(df_null)):
                    df_new.at[df_null.index[i],item]=mode([df_new.iloc[df_numeric.iloc[j].name][item] for j in indices[i] ])
        return df_new
    
    def __bin_filler(self,df_new,vals):
                
        for item in vals: 
            df_null=df_new[pd.isnull(df_new[item])].select_dtypes(np.number)
            if not df_null.empty:
                df_numeric=df_new[~pd.isnull(df_new[item])].select_dtypes(np.number)
                indices = NearestNeighbors(n_neighbors=self.k, algorithm=self.algorithm,n_jobs=self.n_jobs)\
                    .fit(df_numeric).kneighbors(df_null.values,return_distance=False)
                for i in range(len(df_null)):
                    df_new.at[df_null.index[i],item]=mode([df_new.iloc[df_numeric.iloc[j].name][item] for j in indices[i] ])
        return df_new
    
    def __id_filler(self,vals):
        id_list_en=['id','code', 'Unique code','index','indx','seq','no']
        id_list_fa=['کد ملی','کد','کدملی' ,'ملي','کد‌ملی','شماره','یکتا','شناسه یکتا','کد یکتا ']
        for col in vals.columns:
            if type(col)==str:
                for wrd in id_list_en:
                    if wrd in col.lower():
                        if self.verbose>0:
                            vals.set_index(col,inplace=True)  
                            print(f'the {col} column is id so we set it to  index ') 
                            print("***************************************************")
                            
                for wrd in id_list_fa:
                    if wrd in col:
                        vals.set_index(col,inplace=True)
                        if self.verbose>0:
                            print(f'the {col} column is id so we set it to  index ') 
                        print("***************************************************")
        return vals
    
    def __int_filler(self,df_new,vals):
        for item in vals:
            df_null=df_new[pd.isnull(df_new[item])].select_dtypes(np.number).drop(item,1).fillna((df_new.mean()))
            if not df_null.empty:
                df_numeric=df_new[~pd.isnull(df_new[item])].select_dtypes(np.number).drop(item,1).fillna((df_new.mean()))
                indices = NearestNeighbors(n_neighbors=self.k, algorithm=self.algorithm,n_jobs=self.n_jobs)\
                    .fit(df_numeric).kneighbors(df_null.values,return_distance=False)
                for i in range(len(df_null)):
                    df_new.at[df_null.index[i],item]=mode([df_new.iloc[df_numeric.iloc[j].name][item] for j in indices[i] ])
        return df_new
    
    def __float_filler(self,df_new,vals):
        for item in vals:
            df_null=df_new[pd.isnull(df_new[item])].select_dtypes(np.number).drop(item,1).fillna(df_new.mean())
            if not df_null.empty:
                df_numeric=df_new[~pd.isnull(df_new[item])].select_dtypes(np.number).drop(item,1).fillna(df_new.mean())
                
                indices = NearestNeighbors(n_neighbors=self.k, algorithm=self.algorithm,n_jobs=self.n_jobs)\
                    .fit(df_numeric).kneighbors(df_null.values,return_distance=False)
                for i in range(len(df_null)):
                    df_new.at[df_null.index[i] ,item ]=mode([df_new.iloc[df_numeric.iloc[j].name][item] for j in indices[i] ])
        return df_new
    
    def tester():
        pass

def test1():
    # df=generator2(10,1000,prcs=[0.01,0.01,0.01,0.01,0.03,0.01,0,0,0,.2])
     
    raw1=list(range(1,100))
    raw2=list(range(10,1000,10))
    raw3=list(np.arange(.1,10,.1))
    raw4=['A','A','B','C','D','E','F','G','H','A','A','B','C','D','E','F','G','H','A','A','B','C','D','E','F','G','H',
          'A','A','B','C','D','E','F','G','H','A','A','B','C','D','E','F','G','H','A','A','B','C','D','E','F','G','H',
          'A','A','B','C','D','E','F','G','H','A','A','B','C','D','E','F','G','H','A','A','B','C','D','E','F','G','H',
          'A','A','B','C','D','E','F','G','H','A','A','B','C','D','E','C','D','E']
    
    
    df = pd.DataFrame()
    
    df['raw1']=raw1
    df['raw2']=raw2
    df['raw3']=raw3
    df['raw4']=raw4
    
    df.iloc[5,0]=None
    df.iloc[6,1]=None
    df.iloc[7,2]=None
    df.iloc[8,3]=None
    #
    t1_start = process_time()
    
    imp=KNN_imputer()
    res1=imp.filler(df)
    print(res1)
    t1_stop = process_time()
    print(t1_stop-t1_start)
   
    
    temp=1

if __name__=='__main__':
    test1()
   
    
    
    
# %%

