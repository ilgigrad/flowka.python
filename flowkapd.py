import pandas as pd
import numpy as np
from copy import copy,deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
from flowkamd import fmd
from flowkacol import fcol
from flowkafunc import *

class fpd:
    "extend dataframe for Flowka"
    
    def __init__(self,df=None,name=''):
        #super(fpd,self).__init__()
        #self._data = df._data
        self._df=df #the pandas.dataframe
        self._name=name #name of the dataset forsome specific transformation / temporary-> find some functions to avoid this variable
        self.fcol=fcol()
        self.fmd=fmd(df,name)
        self._nans=['?','na','nil','nan','null','','none'] #set some values considered as nan
        self._targets=''
        self._version='0.25'
        self._maxdistcat=0.7 # ratio of max distinct values for categorical columns
        self._logs=[]#logs/nbatch of the session
        self._modified={}#columns modifications of the session
        self._dummized={}#columns set to dummy of the session
        self._loadlogs=[]# recalled logs/batchs
        self._loadmodified={}# recalled modified columns
        self._loaddummized={}#recalled dummized columns
        self._silent=False
        self._addndrop=[]
        self._maxshowsamples=2000 #number ox maximum samples for graphic display
        self._set_category()#build a dict of lists of categorical and quantitative columns
        self._clusters=None
        self._changed=pd.DataFrame()
        self._batchmode=False
        self._talk=talk(True)
        _sample=dict(df.sample(n=1))
        for key,value in _sample.items():
            _sample[key]=list(value)
        self.sample=_sample
        _sample=None


    def log(self, string):
        self._logs.append(string)

    def _get_nans(self):
        return copy(self._nans)

    def _set_nans(self,nan_str):
        self.log("_set_nans( {0})".format(nan_str))
        if len(nan_str)==0:
            raise ValueError("nans list should have at least one element")
        else:
            self._nans = nan_str

    nans = property(_get_nans,_set_nans)


    def _get_clusters(self):
        """ in case of segmentation algorithm, give the number of clusters
        """
        return copy(self._clusters)

    def _set_clusters(self,clusters=0):
        """ set the number of clusters for segmentation algorithms
            if clusters  is  zero or None,  the best number of clusters is seek by algorithm
        """
        self.log("_set_clusters( {0})".format(clusters))
        if clusters==0:
            self._clusters = None
        else:
            self._clusters = clusters
        self.fmd.clusters(self._clusters)

    clusters = property(_get_clusters,_set_clusters)


    def _get_maxshowsamples(self):
        """ return the number ox maximum samples for graphic display
        """
        return copy(self._maxshowsamples)

    def _set_maxshowsamples(self,maxshowsamples):
        """ set the number ox maximum samples for graphic display
        """
        self.log("_set_maxshowsamples({0})".format(maxshowsamples))
        if not isfloat(maxshowsamples):
            raise ValueError("float expected for maxim show samples value")
        else:
            self._maxshowsamples = maxshowsamples

    maxshowsamples = property(_get_maxshowsamples,_set_maxshowsamples)


    def _get_maxdistcat(self):
        """ return the ratio of max distinct values allowed for categorical columns
        """
        return copy(self._maxdistcat)

    def _set_maxdistcat(self,maxdistcat):
        """ set the ratio of max distinct values allowed for categorical columns
            maxdistcat: float, max distinct rate for categories
        """
        self.log("_set_maxdistcat( {0})".format(maxdistcat))
        if not isfloat(maxdistcat):
            raise ValueError("float expected for maxdistinct_categorical value")
        else:
            self._maxdistcat = maxdistcat

    maxdistcat = property(_get_maxdistcat,_set_maxdistcat)
    
    def get_version(self):
        version='flowka.fpd version is {0}'.format(self._version)
        return version

    def set_version(self):
        # nothing
        return

    version = property(get_version,set_version)
    
    def _get_name(self):
        return copy(self._name)

    def _set_name(self,name):
        self.log("_set_name(name = '{0}')".format(name))
        if len(str.strip(name))==0:
            raise ValueError("dataset's name should not be empty ")
        else:
            self._name = name
    
    name = property(_get_name,_set_name)

    def _get_targets(self):
        return copy(self._targets)

    def _set_targets(self,targets=''):
        self.log("_set_targets(targets = '{0}')".format(targets))
        if targets is None or targets==np.nan or targets not in self.df.columns:
            self._targets=''
        elif str(type(targets))==list:
            self._targets =targets[0].replace("'",'').strip()
        else:
            self._targets = targets.replace("'",'').strip()
        
        self._talk.talk("targets set to {0}".format(self.targets))
        #### _log to metadata
        self.fmd.targets(self._targets)

    targets = property(_get_targets,_set_targets)
    
    def __get_df(self):
        return self._df

    def __set_df(self,df=None):
        if len(df)==0:
            raise ValueError("dataframe should have at least one element")
        else:
            self._df = df
            #self.fmd=fmd(df,name=self.name)
            self._set_category()# reinititalize quantitative and categorical list if df is changed

    df = property(__get_df,__set_df)

    def save_change(self,column):
        """save column changed or droped with their intial values 
        """
        if isinlist(column,self.df.columns): 
            self._changed[column]=self.df[column]
    
    def restore_change(self,column):
        """restore column changed or droped with their intial values 
        """
        if isinlist(column,self.df.columns): 
            self.df[column]=self._changed[column]
            self._changed.drop(column,axis=1,inplace=True,errors='ignore')


    def has_change(self,column):
        """test if a column has changed values 
        """
        return isinlist(column,self._changed.columns) 

    def restore_df(self):
        """ restore dataframe to its initial values
        """        
        for elem in reversed(self._addndrop):
            if elem[1]=='drop':#recreate dropped columns
                self.df[elem[0]]=self._changed[elem[0]]
                self._changed.drop(elem[0],axis=1,inplace=True,errors='ignore')
            elif elem[1]=='add':#drop added columns
                self.df.drop(elem[0],axis=1,inplace=True,errors='ignore')
        self._addndrop=[]
        for column in self._changed.columns:#changed columns are set to initial set
            self.df[column]=self._changed[column]
            self._changed.drop(column,axis=1,inplace=True,errors='ignore')
    
    def _set_category(self,columns=None):
        """detect list of categorical and quantitative columns before they are droped or transformed into dummies
           column s : list of columns to set to 'category'            
        """
        #set automatic category for object dtypes and add the categories
        if not self.silent_category and columns is not None :
            self.log("_set_category(columns={0})".format(columns))
        columns=unionlist(list(self.df.select_dtypes(include=['object']).columns),columns)
        for col in columns:
            if isinlist(col,list(self.df.columns)):
                self.df[col]=self.df[col].astype('category', errors='ignore')
        #log to metadata 
        self.fmd.category(columns)

    def _get_category(self):
        """get a list of categorical columns before they are transformed into dummies           
        """
        return self.dtypes['category']

    category = property(_get_category,_set_category)

    def _get_continuous(self):
        """get a list of quantitative columns before they are transformed into dummies           
        """
        return self.dtypes['continuous']

    continuous = property(_get_continuous)
    
    def __get_dtypes(self):
        """get a list of categorical and quantitative columns before they are transformed into dummies           
        """
        category=self.df.select_dtypes(include=['category','object']).columns
        continuous=self.df.select_dtypes(exclude=['category','object','bool']).columns
        boolean=self.df.select_dtypes(include=['bool']).columns
        types={'category':list(category),'continuous':list(continuous),}
        return types

    dtypes = property(__get_dtypes)


    def __get_details(self):
        """
        give some detailed informations about
        distinct or nan values in columns
        """
        info=pd.DataFrame({'dtype': self.df.dtypes,
                           'min': self.df.min(),
                           'max': self.df.max(),
                           'mean': self.df.mean(),
                           'stddev': np.sqrt(self.df.var()),
                           '10%': self.df.quantile(0.10,interpolation='lower'),
                           '25%': self.df.quantile(0.25,interpolation='lower'),
                           '50%': self.df.quantile(0.5,interpolation='midpoint'),
                           '75%': self.df.quantile(0.75,interpolation='higher'),
                           '90%': self.df.quantile(0.90,interpolation='higher'),
                           'nan': self.df.isnull().sum(),
                           'count': self.df.count(),
                           'valid': self.df.count()+self.df.isnull().sum(),
                           'distinct': self.df.nunique(),
                           'distratio': self.df.nunique()/(self.df.count()+self.df.isnull().sum())})
        #info.reset_index(inplace=True)
        #info.rename(index=str,columns={'index':'columns'},inplace=True)
        return info.transpose()

    details = property(__get_details) 

    def batch(self,batchlist=None):
        """make batch transformations from a command list
           batchlist: list of batch to execute or file to load
           if batchlist is none, try to load a previous log file
           loud=False silent mode / loud=True toggle to verbose mode
        """
        self._batchmode=False
        if batchlist is None:# if no list of batch command, try to load a file
            self.loadlog()
            batchlist = self._loadlogs
            self._batchmode=True#set if previous column transformation are also used in the batch (_modified and _dummized)
        elif isinstance(batchlist,str):
            self.loadlog(batchlist)
            self._batchmode=True#set if previous column transformation are also used in the batch (_modified and _dummized)
        if batchlist is not None: #if a command list or a batch file loaded
            for xi in batchlist:
                self._talk.talk(xi)
                eval('self.'+xi)
        self._batchmode=False


    def timize(self,column,timeformat='%d%m%Y-%H:%M:%S.%f',timelist=['time'],drop=True):
        """ transform a column into quantiemes of time
        column : name of the column to transform
        timelist : choose one or more period in ['year','month','day','weekday','hour','time']
        timeformat : must fit to column time format '%Y%m%d-%H:%M:%S.%f' or any combinaison
        drop : drop initial column
        eg: time_ize(column='datetime',timelist=['weekday','hour'],drop=True) 
        """
        if not isinlist(column,list(self.df.columns)):
            self._talk.talk("column '{0}' does not belong to the Dataset ".format(column))
            return None
        else:
            self.log("timize(column='{0}',timeformat='{1}',timelist={2},drop={3})".format(column,timeformat,timelist,drop))
        timedict={
            'year':(self.df[column].apply(lambda x : pd.to_datetime(x,format=timeformat))).dt.year,
            'month':(self.df[column].apply(lambda x : pd.to_datetime(x,format=timeformat))).dt.month,
            'weekday':(self.df[column].apply(lambda x : pd.to_datetime(x,format=timeformat))).dt.weekday,
            'day':(self.df[column].apply(lambda x : pd.to_datetime(x,format=timeformat))).dt.day,
            'hour':(self.df[column].apply(lambda x : pd.to_datetime(x,format=timeformat))).dt.hour,
            'time':(self.df[column].apply(lambda x : pd.to_datetime(x,format=timeformat))).dt.time
        }
        for col in timelist:
            newquantieme=timedict[col]
            self.add=newquantieme.to_frame(column+'_'+col)
        if drop==True:
            self.silent_drop=column
        self.fmd.transform([column])

    def splitize(self, column, separ=' ',keeplist=[0],itemnames=None,drop=True ):
        """split column and keep some items 
           column : name of the column to split
           separ : separator; default is space it can be any char ',' or '-'...
           keeplist : list of items to keep
           itemnames : name of items to replace with
           drop : drop initial column after transform
        """
        if not isinlist(column,list(self.df.columns)):
            self._talk.talk("column '{0}' does not belong to the Dataset ".format(column))
            return None
        else:
            self.log("splitize(column='{0}',separ='{1}',keeplist={2},itemnames={3},drop={4})".format(column,separ,keeplist,nnone(itemnames),drop))
        
        tempcol=self.df[column]
        if drop==True:
            self.silent_drop=column
        for i, keepitem in enumerate(keeplist):
            newsplit=tempcol.apply(lambda x : x.split(separ)[keepitem] if len(x.split(separ))>keepitem else None)
            try: 
               newsplitname=itemnames[i]
            except:
                newsplitname=column+'_'+str(i)
            self.add=newsplit.to_frame(newsplitname)
        
        self.fmd.transform([column])


    def substrize(self, column, splitpos,keeplist=[0],itemnames=None,drop=True ):
        """split base on character position
           column : name of the column to split
           splitpos : position of split characters 'V1M66' [2,3] -> ['V1','M','66'] 
           keeplist : list of items to keep
           itemnames : name of items to replace with
           drop : drop initial column after transform
        """
        if not isinlist(column,list(self.df.columns)):
            self._talk.talk("column '{0}' does not belong to the Dataset ".format(column))
            return None
        else:
            self.log("substrize(column='{0}',splitpos={1},keeplist={2},itemnames={3},drop={4})".format(column,splitpos,keeplist,nnone(itemnames),drop))
        if splitpos[0]!=0:
            splitpos[0:0]=[0]
        splitpos.append(None)
        tempcol=self.df[column]
        if drop==True:
            self.silent_drop=column
        for i, keepitem in enumerate(keeplist):
            newsplit=tempcol.apply(lambda x : x[splitpos[keepitem]:splitpos[keepitem+1]])
            try: 
               newsplitname=itemnames[i]
            except:
                newsplitname=column+'_'+str(i)
            self.add=newsplit.to_frame(newsplitname)
        self.fmd.transform([column])

    def regexize(self, column, regexstr,keeplist=[0],itemnames=None,drop=True ):
        """split base on regex expression
           column : name of the column to split
           regexstr : regex string
           keeplist : list of items to keep
           itemnames : name of items to replace with
           drop : drop initial column after transform
        """
        if not isinlist(column,list(self.df.columns)):
            self._talk.talk("column '{0}' does not belong to the Dataset ".format(column))
            return None
        else:
            self.log("regexize(column='{0}',regexstr='{1}',keeplist={2},itemnames={3},drop={4})".format(column,regexstr,keeplist,nnone(itemnames),drop))
        import re
        tempcol=self.df[column]
        if drop==True:
            self.silent_drop=column
        for i, keepitem in enumerate(keeplist):
            newsplit=tempcol.apply(lambda x : re.match(regexstr,x).group(keepitem))
            try:
               newsplitname=itemnames[i]
            except:
                newsplitname=column+'_'+str(i)
            self.add=newsplit.to_frame(newsplitname)
        self.fmd.transform([column])


    def unknowns(self):
        """replace with nan for unknown values
            nans is define in the class : ['?','na','nil','nan','null','']
            type df.nans for current list of nans string
        """
        self.log("unknowns()")
        
        for i in range(self.df.shape[1]):
            self.df[self.df.columns[i]]=self.df[self.df.columns[i]].apply(lambda x : np.nan if str.lower(str(x)) in self._nans else x)
    

    def automatic_drop(self):
        """determine and drop categorical columns with too-much distinct values 
           eg: ids, names or descriptions rather than categories
           maxdistinct_categorical : set the ratio of max distinct values allowed for categorical columns 
        """
        details=self.details.transpose()
        # columns with distinct values are less than 2 (nan or 1) -> single value = allways the same
        presume_unique=list(details[(details.distinct<2)].index)
        # 'object' columns with a ration of distinct values on counted values greater than the madistinct_categorical ratio (eg names, descriptions)
        presume_desc=list(details[(details.dtype=='category') & (details.distratio>self._maxdistcat)].index)
        # 'int64' columns with as much distinct values as counted values (eg ids, rank)
        presume_id=list(details[(details.dtype=='int64') & (details.distratio==1)].index)
        details=None
        drop=list(presume_unique)+list(presume_desc)+list(presume_id)
        return drop

    def _set_drop(self,columns):
        """drop columns listed in columns 
        """
        self._talk.talk("drop:{0}".format(columns))
        if type(columns)!=list:
            columns=[str(columns)]
        if not self.silent_drop:
            self.log("_set_drop({0})".format(columns))
        if isinlist(self.targets,columns):
            self.targets=''
        for col in columns:
            if col in self.df.columns:
                self._addndrop.append((col,'drop'))
                self.save_change(col)
                self.df.drop(col,axis=1,inplace=True,errors='ignore')#drop columns from the main dataframe
                #log tometadata
                if not self.silent_drop:
                    self.fmd.drop([col])
    
    def _get_drop(self):
        """return all droped columns 
        """
        drops=[]
        for dropped in self._addndrop:
            if dropped[1]=='drop':
                drops.append(dropped[0])
        return drops

    drop = property(_get_drop,_set_drop)

    #silent_drop is created to stop log and metadata outputs while dropping columns
    #when drop is called from a class function and not from an outside/user called 

    def __set_silent_drop(self,columns):
        """internal class function that prevent to log information from drop
           disactive the log/metadata mode and run drop function
        """
        self._silent=True
        self.drop=columns
        self._silent=False

    def __get_silent_drop(self):
        """internal class function that prevent to log information from drop
           return the status of silent_drop mode (True/Flase)  
        """
        return self._silent

    silent_drop = property(__get_silent_drop,__set_silent_drop)

    def __set_silent_category(self,columns):
        """internal class function that prevent to log information from category set
           disactive the log/metadata mode and run category function
        """
        self._silent=True
        self.category=columns
        self._silent=False

    def __get_silent_category(self):
        """internal class function that prevent to log information from category set
           return the status of silent_category mode (True/Flase)  
        """
        return self._silent

    silent_category = property(__get_silent_category,__set_silent_category)


    def _set_add(self,dataframe):        
        """add columns to the dataset
           must be a dataframe 
        """
        self._talk.talk("add:{0}".format(list(dataframe.columns)))
        self.df=pd.concat([self.df,dataframe],axis=1,)
        column=list(dataframe.columns)[0]
        #self._adds=unionlist(self._adds,list(column))
        self._addndrop.append((column,'add'))

    def _get_add(self):
        """return all added columns 
        """
        adds=[]
        for addeded in self._addndrop:
            if added[1]=='add':
                adds.append(added[0])
        return adds

    add = property (_get_add,_set_add)


    def missing_values(self):
        """give a table of missing values/nans for each columns
        """
        nmv= self.df.count() #non missing values
        mv = self.df.isnull().sum() #missing values
        mvrate = 100 * self.df.isnull().sum() / len(self.df) #missing values rate
        mv_table = pd.concat([nmv,mv,mvrate],axis=1)
        mv_table = mv_table.rename(
        columns = {0 : 'Non_Missing_Values',1 : 'Missing_Values',2 : '% of Total Values'})
        mv_table=mv_table[mv_table.Missing_Values != 0]
        return mv_table
    
    def nan_replace(self,mode='mean',columns=None, value=0):
        """replace nan values with a function of other values in the same column
           mode : ['mean','median','value','mostfreq', 'lessfreq','max','min']
           columns : if None, applied to all Nan values of all columns
           value : if mode 'value' is set then a value to replace nan values is needed

           'mean': mean 
           'median': quantile 50%
           'value' : a specific value (numeric or category)
           'mostfreq' : most frequent value of the column (different from median)
           'lessfreq' : less frequent value of the column (different from last quantile)
           'max' : maximum value
           'min' : minimum value
           ['mean','median','max','min'] are not allowed to categorical variable
           ['value','mostfreq', 'lessfreq'] can be applied to categorical
        """
        
        if not isinlist(columns,list(self.df.columns)):
            self._talk.talk("column '{0}' does not belong to the Dataset ".format(columns))
            return None
        else:
            self.log("nan_replace(mode='{0}',columns={1}, value={2})".format(mode,nnone(columns), value))
        
        columns = [columns,list(self.df.columns)][columns is None]
        for column in columns:
            if mode=='median':
                self.df[column].fillna(self.df[column].quantile(0.5,interpolation='midpoint'),inplace=True)
            elif mode=='mean':
                self.df[column].fillna(self.df[column].mean(),inplace=True)
            elif mode=='value':
                if self.df[column].dtypes.kind=='O' and value not in list(set(self.df[column].astype("category"))):
                    self.df[column]=self.df[column].cat.add_categories(value)
                self.df[column].fillna(value,inplace=True)
            elif mode=='mostfreq':
                self.df[column].fillna(list(self.df[column].value_counts().head(1).reset_index()['index'])[0],inplace=True)
            elif mode=='lessfreq':
                self.df[column].fillna(list(self.df[column].value_counts().tail(1).reset_index()['index'])[0],inplace=True)
            elif mode=='max':
                self.df[column].fillna(self.df[column].max(),inplace=True)
            elif mode=='min':
                self.df[column].fillna(self.df[column].min(),inplace=True)
            
    def intize(self,columns):
        """cast values of a column into int
        """
        if not isinlist(columns,list(self.df.columns)):
            self._talk.talk("column '{0}' does not belong to the Dataset ".format(columns))
            return None
        else:
            self.log("intize(columns={0})".format(nnone(columns)))

        for column in columns:
            #self.df[column]=self._df[column].apply(lambda x: int(x) if str(x).isdigit() else x)
            self.df[column]=self.df[column].astype(int, errors='ignore')
            self.fmd.continuous([column])

    def floatize(self,columns):
        """cast values of a column into int
        """
        if not isinlist(columns,list(self.df.columns)):
            self._talk.talk("column '{0}' does not belong to the Dataset ".format(columns))
            return None
        else:
            self.log("floatize(columns={0})".format(nnone(columns)))
        for column in columns:
            self.df[column]=self._df[column].apply(lambda x: float(x) if str(x).isdigit() else x)
            self.df[column]=self.df[column].astype(float, errors='ignore')
            self.fmd.continuous([column])

    def boolnotnans(self,columns,values=[0,1]):
        """set 1 for every not nan values and 0 for nans,
           columns: list of columns to transform
           values : value for nan, value for not nan   
        """
        if not isinlist(columns,list(self.df.columns)):
            self._talk.talk("column '{0}' does not belong to the Dataset ".format(columns))
            return None
        else:
            self.log("boolnotnans(columns={0},values={1})".format(nnone(columns),values))
        for column in columns:
            for value in values:
                self.save_change(column)
                #add the new values in distinct values of the categorical column (to avoid pandas errors)
                if self.df[column].dtypes.kind=='O' and value not in list(set(self.df[column].astype("category"))):
                    self.df[column]=self.df[column].cat.add_categories(value)
            #then replace na by first value of the list given in function's parameters
            self.df[column]=self.df[column].fillna(value=values[0])         
            #the others values are set to the second value of the list given in the function's parameters 
            self.df[column]=self.df[column].apply(lambda x : x if x==values[0] else values[1])
            #modify category and continous list / type of column
            if self.df[column].dtypes.kind=='O':
                self.df[column]=self.df[column].astype("category", errors='ignore')
                self.fmd.continuous([column])
            else:
                self.df[column]=self.df[column].astype(self.df[column].dtypes, errors='ignore')
                self.fmd.category([column]) 

    def drop_allnans(self):
        """drop columns with all values equalto nan
        """
        self.log("drop_allnans()")
        #self.df.dropna(axis=1,how='all',inplace=True)
        na=list(set(self.df.columns)-set(self.df.dropna(axis=1,how='all',inplace=False).columns))
        self.silent_drop=na

    def drop_anynans(self):
        """drop rowss with any values equal to nan 
            -- if df.nan_to_mean() as been run before df.drop_anynans(),
            -- then only rows with categorical nans will be removed
        """
        self.log("drop_anynans()")
        self.df.dropna(axis=0,how='any',inplace=True)

    def set_frequencies(self,columns=None):
        """replace categories by frequency of each category
           by default object / catgorical columns are transformed into dummies;
           by using set_frequencies, categorical are replaced by their frequencies
           freqcol : limit the transformation to selected columns   
           if df =['A','B','C','A','A','B'], 
           it will be replace by df' =[3,2,1,3,3,2] 
        """
        if columns is None:
            columns=self.dtypes['category']
        if not isinlist(columns,list(self.df.columns)):
            self._talk.talk("column {0} does not belong to the Dataset ".format(columns))
            return None
        else: 
            self.log("set_frequencies(columns={0})".format(nnone(columns)))
        self._talk.talk("frequencies : {0} ".format(columns))

        for column in columns:
            function="(((self.df[column].value_counts())/(self.df[column].value_counts()).sum())).to_dict()" 
            self.transform(column,function)
        #log to metadata
        self.fmd.frequencies(columns)   
        return columns   

    def set_dummies(self,columns=None):
        """replace categories by dummies columns
           by default object / catgorical columns are transformed into dummies; numerical columns (age,level...) can be also transformed into dummies if they have been set to category type 
           columns = ['country','age','targets']
           by default only 'country' and will be transformed into dummy.
           if ['age'] as been set to category then ['country','age'] will be transformed into dummies  
        """
        if columns is None:
            columns=self.dtypes['category']
        elif not isinlist(columns,list(self.df.columns)):
            self._talk.talk("column {0} does not belong to the Dataset ".format(columns))
            return None
        else:
            self.silent_category=columns
        self.log("set_dummies(columns={0})".format(nnone(columns)))
        #self.df=pd.get_dummies(self.df,columns=columns,drop_first=True)
        for column in columns:
            categories=list(set(self.df[column].cat.categories)-set([None,np.nan]))
            if self._batchmode==True:#batch mode
                categories=nvl(self._loaddummized[column],categories)
            self._dummized[column]=categories
            for subcol in categories:
                newcolumn=self.df[column].apply(lambda x : 1 if ~pd.isnull(x) and x==subcol else 0)
                self.add=newcolumn.to_frame(column+'_'+str(subcol))
                if column==self._targets:#if targets was in dummy column take the first value of column targets as new targets
                    self.targets=column+'_'+str(subcol)#do it only one time (targets name is then updated)
            self.silent_drop=[column]    
        self.fmd.dummies(columns)   
        return columns

    def set_ordinals(self,column,categories=None):
        """replace categories by ordered numbers 
           case of literal ranking (good/intermediate/bad) 
           The order depend on a ordered list of all categorical values given in categories
           if the value is not present is the categories, it is set to 0 (zero)
           column: categorical column to transform
           categories: list of ordered values

           eg: df=[foo,bar,apple,dog,bar,foo,cat] and categories=[bar,foo,cat] -> df=[2,1,0,0,1,2,3]  
        """
        if not isinlist(column,list(self.df.columns)):
            self._talk.talk("column '{0}' does not belong to the Dataset ".format(column))
            return None
        else:
            self.log("set_ordinals(column='{0}',categories={1})".format(nnone(column),nnone(categories)))
        
        #transform categories into dict
        self.silent_category=[column]
        if categories is None:
            categories=list(self.df[column].cat.categories)
        #create a column that counts each value and  counts the number of all value and calculate frequencies (counts(item-i)/sum(all items))
        categoriesdict="dict([key, i] for i,key in enumerate("+str(categories)+"))"
        self.transform(column=column,function=categoriesdict)

    def transform(self,column, function):
        if self._batchmode==True:
            if self.has_change(column):
                self.restore_change(column)# prevent from doing batch twice on the same column
            transform= nvl(self._loadmodified.get(column),eval(function))
        else:
            transform= eval(function)
        self._modified[column]=transform
        self.save_change(column)
        self.df[column]=self.df[column].apply(lambda x: self._modified[column][x])
        return self

    def unique(self,column):
        """give distinct/unique values of a column
        """
        if isinlist(column,list(self.df.columns)):
            return list(self.df[column].unique())
        else:
            self._talk.talk("column '{0}' does not belong to the Dataset ".format(column))


    def head(self,rows=5):
        """pd.df.head() function
        """ 
        return self.df.head(rows)

    def describe(self):
        """pd.df.describe() function
        """
        return self.df.describe()

    def info(self):
        """pd.df.info() function
        """
        return self.df.info()


    def pairplot(self,columns=None):
        """display a pairplot graph of all numerical/contiuous columns
           pairplot shows correlations between variables wether they are linear or not
           columns= list of columns to display to bypass default selection
        """
        # prevent large pairplot of boolean columns/ booleans are only used for 'hue' parameter as a targets
        #tempdf=self.df.sample(n=self.maxshowsamples)
        tempdf=self.df.sample(n=self._maxshowsamples,replace=True)
        targets=[self.targets,None] [len(self.targets)==0]
        if columns is None:
            details=self.details.transpose()
            columns=intersectionlist(self.dtypes['continuous'],list(details[details['distinct']>2].index))
            details=None
        else:
            columns=intersectionlist(columns,list(tempdf.columns))
        if targets is not None and not isinlist([targets],columns):
            columns.append(targets)
        g = sns.PairGrid(tempdf[columns], hue=targets,dropna=True,palette=self.fcol.palette)
        g = g.map_diag(plt.hist,bins=20)
        g = g.map_offdiag(plt.scatter)
        if targets is not None and tempdf[targets].nunique()<6: #do no build a colorization/hue of targets column if more than 5 distinct values 
            g = g.map_lower(sns.kdeplot,cmap=self.fcol.cmap)
            g = g.add_legend()
        plt.show()


    def lmplot(self,columns=None):
        """display a lmplot graph of all numerical columns
           by default it displays only columns with valratio > 0.2
           columns: force a list of columns to display; those columns have to be continuous 
        """
        import itertools as it
        #tempdf=self.df.sample(n=self.maxshowsamples)
        tempdf=self.df.sample(n=self._maxshowsamples,replace=True)
        targets=[self.targets,None] [len(self.targets)==0]
        if columns is None:
            details=self.details.transpose()
            columns=intersectionlist(self.dtypes['continuous'],list(details[details['distratio']>0.05].index))
            #columns=self.dtypes['continuous']
            details=None
        else:
            columns=intersectionlist(columns,self.dtypes['continuous'])
        #do no build a colorization/hue of targets column if more than 5 distinct values 
        if  targets is not None and tempdf[targets].nunique()<6:
            huelegend= True
            try:
                columns.remove(targets)
            except:
                False
        else:
            huelegend= False
        for lm in it.combinations(columns,2):
                ax=sns.lmplot(x=lm[0],y=lm[1],data=tempdf,hue=targets,size=6,aspect=1,fit_reg=False,palette=self.fcol.palette, legend=huelegend)
                ax=sns.kdeplot(tempdf[lm[0]],tempdf[lm[1]],zorder=0,n_levels=10,shade=True,cmap=self.fcol.cmap)

   
    def heatmap(self,columns=None):
        """display a heatmap all columns
           columns : for a specific list of columns to set in correlation map
           heatmap point out only linear correlations / it does not show non linear correlation wether they are monotonic or not
        """
        #tempdf=self.df.sample(n=self.maxshowsamples)
        #tempdf=self.df.sample(n=self._maxshowsamples)
        tempdf=self.df
        if columns is None:
            columns = tempdf.columns
        plt.figure(figsize=(10,10))
        sns.heatmap(tempdf[columns].corr(method='pearson'),cmap=self.fcol.cmap)
        plt.title(self._name)

    def histmap(self,columns=None):
        """display a distribution map of each columns
        """
        targets=[self.targets,None] [len(self.targets)==0]
        if columns is None:
            columns = list(self.df.columns)
        
        if targets is not None and (self.df[targets].nunique()==2):
            try:
                columns.remove(targets)
            except:
                False
            for element in columns:
                plt.figure(figsize=(10,10))
                color=self.fcol.multicolor(2)
                self.df[self.df[targets]==self.details.loc['min',self.targets]][element].hist(alpha=0.7,
                    color=color[0],bins=30,
                    label=targets+'='+str(self.details.loc['min',targets]),
                    edgecolor=self.fcol.c())
                self.df[self.df[targets]==self.details.loc['max',targets]][element].hist(alpha=0.7,
                    color=color[1],bins=30,
                    label=targets+'='+str(self.details.loc['max',targets]),
                    edgecolor=self.fcol.c())
                plt.legend()
                plt.xlabel(element)
        else:
            for element in columns:
                g = sns.FacetGrid(self.df,size=4,aspect=2)
                g.map(plt.hist,element,bins=30,edgecolor=self.fcol.c(),color=self.fcol.color)
        plt.title(self._name)

    def save(self,pdfile=None,mode='csv'):
        """export dataframe to 'csv' or to 'dat'
           by default (if fpdfile is None, exported to flowka_*datsetname)
        """
        import pickle
        pdfile=(str(pdfile).strip(),'./work/flowka_'+self.name)[pdfile is None]+('.dat','.csv')[mode=='csv']
        pickle.dump(self, open(pdfile, 'wb')) if mode=='dat' else self.df.to_csv(open(pdfile, 'w'), header=True,index=False)
        print ("data exported to {0}".format(pdfile))

    def savelog(self,logfile=None):
        """export logs to a file for later use
          logfile : set a specific log file; by default it is set to local 'flowka_log_*name.dat'
        """
        import pickle
        if logfile is None:
            logfile = './work/flowka_log_'+self.name+'.dat'
        else:
            logfile=str(logfile).strip()
        try:
            saves={'logs':self._logs,'modified':self._modified,'dummized':self._dummized}
            pickle.dump(saves, open(logfile, 'wb'))
        except ValueError:
            print("unable to save logs {0}".format(logfile))

    def loadlog(self,logfile=None):
        """export logs to a file for later use
           logfile : set a specific log file; by default it is set to local  'flowka_log_*name.dat'
        """
        import pickle

        if logfile is None:
            logfile = './work/flowka_log_'+self.name+'.dat'
        else:
            logfile=str(logfile).strip()
        try:
            loads = pickle.load(open(logfile, 'rb'))
            self._loadmodified=loads['modified']
            self._loaddummized=loads['dummized']
            self._loadlogs=loads['logs']
        except:
            loads=None
            print("logs {0} not found".format(logfile))
        
        return loads

    def sample_template(self):
        """give a structure for a manual prediction
           copy, paste, modify and execute the following sequence of code
        """
        xdict=self.sample
        print("manual={")
        for i,key in enumerate(xdict):
            comma=[' ',','][i<len(xdict)-1]
            print("\'{0}\': {1}{2}".format(key,xdict[key],comma))
        print("}") 
  

