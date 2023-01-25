#%%
import pandas as pd
import numpy as np
from generetor import *
from sklearn.preprocessing import MinMaxScaler
from nearest_neighbor import n_neighbors
from statistics import *
from time import process_time
import warnings

#%%vrb_level=1


class KNN_imputer:
    def __init__(self,trsh=.05,vrb_level=1):
        
        self.verbose=vrb_level
        self.missing_treshold=trsh

        warnings.filterwarnings('ignore')
    
    def __type_detector(self,vals,A_dtypes):
        if not vals.empty:
                    categorical=[]
                    integer_col =[]
                    float_col =[]
                    for item in A_dtypes:
                        if A_dtypes[item].type==str or A_dtypes[item].type==np.bool_  :
                            categorical.append(item)

                        elif  A_dtypes[item].type==np.intp:
                            integer_col.append(item)
                        elif  A_dtypes[item].type==np.float_:
                            float_col.append(item)
                   
                    return categorical,float_col,integer_col
    
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
            categorical,float_col,integer_col=self.__type_detector(df_new,A_dtypes)
            df_scale=df_new.copy()
            df_scale[df_scale.select_dtypes(np.number).columns]=pd.DataFrame(MinMaxScaler().fit_transform(X=df_scale[df_scale.select_dtypes(np.number).columns]),columns=df_scale.select_dtypes(np.number).columns)

            if  integer_col:
                df_new=self.__filler(df_scale,df_new,integer_col,"int")
            if  float_col:
                df_new=self.__filler(df_scale,df_new,float_col,'float')
            if  categorical:
                df_new=self.__filler(df_scale,df_new,categorical,'cat')
            df_new=self.__id_filler(df_new)
           
            
            return df_new      
        else:
            print('all dataframe had been deleted due to missing treshhold')
    
    
    

        
    def __filler(self,df_scale,df_new,vals,check):
        for item in vals:
            print(item)
            if df_new[item].isnull().sum()==0:
                continue
            df_null=df_scale[pd.isnull(df_scale[item])].drop(item,1).reset_index(drop=True)
            for col in df_null.select_dtypes(include=np.number):
                df_null[col].fillna(df_null[col].mean(),inplace=True)
            for col in df_null.select_dtypes(exclude=np.number):
                df_null[col].fillna(df_null[col].mode().iloc[0],inplace=True)
        
             
                
                
            if not df_null.empty:                
                for count in range(len(df_null.values)):

                    
                    
                    df_not_null=df_new[~pd.isnull(df_new[item])].reset_index(drop=True)
                    for col in df_not_null.select_dtypes(include=np.number):
                        df_not_null[col].fillna(df_not_null[col].mean(),inplace=True)
                    for col in df_not_null.select_dtypes(exclude=np.number):
                        df_not_null[col].fillna(df_not_null[col].mode().iloc[0],inplace=True)

                    if check=='int':
                        df_new.at[df_null.index[count],item]=int(mean([df_not_null.iloc[j][item] for j in (n_neighbors(normalization=True,return_distance=False)\
                                    .get_neighbors(df_not_null,df_null.values[count]))]))
         
                        
                    elif check=="float":
                        
                        df_new.at[df_null.index[count],item]=mean([df_not_null.iloc[j][item] for j in (n_neighbors(normalization=True,return_distance=False)\
                                    .get_neighbors(df_not_null,df_null.values[count]))])
                        
                    elif check=='cat':

                        
    
                        df_new.at[df_null.index[count],item]=mode([df_not_null.iloc[j][item] for j in (n_neighbors(normalization=True,return_distance=False)\
                                    .get_neighbors(df_not_null,df_null.values[count][0]))]) 
                    
        return df_new
    def __id_filler(self,vals):
        id_list_en=['id','code', 'Unique code','index','indx','seq','no']
        id_list_fa=['کد ملی','کد','کدملی' ,'ملي','کد‌ملی','شماره','یکتا','شناسه یکتا','کد یکتا ']
        for col in vals.columns:
                for wrd in id_list_en:
                    if col.lower().find(wrd)!=-1:
                        vals.set_index(col,inplace=True)  
                        
                        if self.verbose>0:
                            print(f'the {col} column is id so we set it to  index ') 
                            print("***************************************************")
                            
                for wrd in id_list_fa:
                    if col.find(wrd)!=-1:
                        vals.set_index(col,inplace=True)
                        if self.verbose>0:
                            print(f'the {col} column is id so we set it to  index ') 
                        print("***************************************************")
        return vals
    
    def tester():
        pass

def test1():
    t1_start = process_time()
    df=pd.read_csv(r'train(1).csv')
    imp=KNN_imputer()
    res1=imp.filler(df)
    print(res1)
    t1_stop = process_time()
    print("Elapsed time during the whole program in seconds:",
										t1_stop-t1_start)



def df_gen(cat_vals=['A','A','B','C','D','E','F','G','H','A','A','B','B','A','D','G','F','G','H','A']):
    
    # df=generator2(10,1000,prcs=[0.01,0.01,0.01,0.01,0.03,0.01,0,0,0,.2])
     
    raw1=cat_vals#list(range(1,21))
    raw2=list(range(10,2000,100))
    raw3=list(np.arange(.1,20,1))
    raw4=cat_vals
    df = pd.DataFrame()
    

    df['raw1']=raw1
    df['raw2']=raw2
    df['raw3']=raw3
    df['raw4']=raw4

    df.iloc[5,0]=None
    df.iloc[6,1]=None
    df.iloc[7,2]=None
    df.iloc[8,3]=None
    print(df)

    return df

def test2():
    #
    cat_vals_box=[
        
     [False,True,True,False,True,False,True,False,False,True,True,True,False,False,True,True,True,False,True,False],

     
     ]
    
    for cat_vals in cat_vals_box:
        t1_start = process_time()
        
        df=df_gen(cat_vals)
      
        imp=KNN_imputer()
        res1=imp.filler(df)
        print(res1)
     
        t1_stop = process_time()
        print("Elapsed time during the whole program in seconds:",
										t1_stop-t1_start)
    
def test3():
    t1_start = process_time()
    df=generator2(4,10,[3,4],prcs=[0.01,.02,.03,.04])

    imp=KNN_imputer()
    res1=imp.filler(df)
    t2_start = process_time()
    print("Elapsed time during the whole program in seconds:",
										t2_start-t1_start)
    print(res1)
if __name__=='__main__':
    test1()
   
    
    
    
# %%

