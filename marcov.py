import time
import pandas as pd
import numpy as np
import collections
from itertools import chain
import itertools
from scipy.stats import stats
import statistics


def unique(list1):  
    unique_list = []   
    for x in list1: 
        if x not in unique_list: 
            unique_list.append(x) 
        
    return(unique_list)


def split_fun(path):
    return path.split('>')


def calculate_rank(vector):
  a={}
  rank=0
  for num in sorted(vector):
    if num not in a:
      a[num]=rank
      rank=rank+1
  return[a[i] for i in vector]


def transition_matrix_func(import_data):
    
    z_import_data=import_data.copy()
     
    z_import_data['path1']='start>'+z_import_data['path']
    z_import_data['path2']=z_import_data['path1']+'>convert'
    
    
    z_import_data['pair']=z_import_data['path2'].apply(split_fun)
    
    zlist=z_import_data['pair'].tolist()
    zlist=list(chain.from_iterable(zlist))
    zlist=list(map(str.strip, zlist))
    T=calculate_rank(zlist)
    
    M = [[0]*len(unique(zlist)) for _ in range(len(unique(zlist)))]
    
    for (i,j) in zip(T,T[1:]):
        M[i][j] += 1
    
    x_df=pd.DataFrame(M)
        
    np.fill_diagonal(x_df.values,0)
    
    x_df=pd.DataFrame(x_df.values/x_df.values.sum(axis=1)[:,None])
    x_df.columns=sorted(unique(zlist))
    x_df['index']=sorted(unique(zlist))
    x_df.set_index("index", inplace = True) 
    x_df.loc['convert',:]=0
    return(x_df)


def simulation(trans,n):
   
    sim = ['']*n
    sim[0] = 'start'
    i = 1
    while i < n :
        sim[i] = np.random.choice(trans.columns, 1, p=trans.loc[sim[i-1],:])[0]
        if sim[i] == 'convert':
            break
        i = i+1
        
    return sim[0:i+1]


def markov_chain(data_set, no_iteration=10, no_of_simulation=10000, alpha=5):

    import_dataset_v1 = data_set.copy()
    import_dataset_v1 = (import_dataset_v1.reindex(import_dataset_v1.index.repeat(import_dataset_v1.conversions))).reset_index()
    # print(import_dataset_v1)
    import_dataset_v1['conversions'] = 1

    import_dataset_v1 = import_dataset_v1[['path','conversions']]
    
    import_dataset = (import_dataset_v1.groupby(['path']).sum()).reset_index()
    import_dataset['probability'] = import_dataset['conversions']/import_dataset['conversions'].sum()
    
    final = pd.DataFrame()

    for k in range(0, no_iteration):
        start = time.time()
        import_data=pd.DataFrame({'path':np.random.choice(import_dataset['path'],size=import_dataset['conversions'].sum(),p=import_dataset['probability'],replace=True)})
        import_data['conversions']=1                           
    
        tr_matrix=transition_matrix_func(import_data)
        channel_only = list(filter(lambda k0: k0 not in ['start','convert'], tr_matrix.columns)) 
    
        ga_ex=pd.DataFrame()
        print(ga_ex)
        tr_mat=tr_matrix.copy()
        p=[]
        
        i=0
        while i<no_of_simulation:
            p.append(unique(simulation(tr_mat,1000)))
            i=i+1
           
        
        path=list(itertools.chain.from_iterable(p))
        counter=collections.Counter(path)
        
        df=pd.DataFrame({'path':list(counter.keys()),'count':list(counter.values())})
        df=df[['path','count']]
        ga_ex=ga_ex.append(df,ignore_index=True) 
        
        df1=(pd.DataFrame(ga_ex.groupby(['path'])[['count']].sum())).reset_index()
        
        df1['removal_effects']=df1['count']/len(path)
        #df1['removal_effects']=df1['count']/sum(df1['count'][df1['path']=='convert'])
        df1=df1[df1['path'].isin(channel_only)]
        df1['ass_conversion']=df1['removal_effects']/sum(df1['removal_effects'])
                   
        df1['ass_conversion']=df1['ass_conversion']*sum(import_dataset['conversions']) 
        
        final=final.append(df1,ignore_index=True)
        end = time.time()
        t1=(end - start)
        print(t1)   
    
    '''
    H0: u=0
    H1: u>0
    '''


    unique_channel=unique(final['path'])
    #final=(pd.DataFrame(final.groupby(['path'])[['ass_conversion']].mean())).reset_index()
    final_df=pd.DataFrame()
    
    for i in range(0,len(unique_channel)):
        
        x=(final['ass_conversion'][final['path']==unique_channel[i]]).values
        final_df.loc[i,0]=unique_channel[i]
        final_df.loc[i,1]=x.mean()
        
        v=stats.ttest_1samp(x,0)
        final_df.loc[i,2]=v[1]/2
        
        if v[1]/2<=alpha/100:
            final_df.loc[i,3]=str(100-alpha)+'% statistically confidence'
        else:
            final_df.loc[i,3]=str(100-alpha)+'% statistically not confidence'
        
        final_df.loc[i,4]=len(x)
        final_df.loc[i,5]=statistics.stdev(x)
        final_df.loc[i,6]=v[0]
        
    final_df.columns=['channel','ass_conversion','p_value','confidence_status','frequency','standard_deviation','t_statistics']       
    final_df['ass_conversion']=sum(import_dataset['conversions']) *final_df['ass_conversion'] /sum(final_df['ass_conversion'])
    
    return final_df,final


# import the channel attribution example csv
import_dataset = pd.read_csv('channel attribution example - sheet1.csv')
print(import_dataset)
data, dataset = markov_chain(import_dataset, no_iteration=10, no_of_simulation=10000, alpha=5)
