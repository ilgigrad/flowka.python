import pandas as pd
import numpy as np
from copy import copy,deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
from flowkacol import fcol
from flowkafunc import isinlist, isfloat,unionlist,intersectionlist,nnone,nvl
from flowkapd import fpd
from sklearn.model_selection import train_test_split
sns.set_style('whitegrid')

class fmo:
    "model for Flowka"
    def __init__(self,ds=None):
        self._ds=ds #the Flowka dataframe extension
        self._model=None
        self._knn_max=10000 #maximum values for knn algorithm
        self._kmns_max=10 #maximum values for knn algorithm
        self.Xy()
        self._test_size=0.3
        self._algo=None
        self._scaler=None

    def _set_test_size(self,test_size=0.3):
        """set the ratio of the test  and the train sets
        """
        self._test_size=test_size


    def _get_test_size(self):
        """get the ratio of the test  and the train sets
        """
        return self._test_size

    test_size=property(_get_test_size,_set_test_size)

    def Xy(self):
        """create a X and a y dataframe 
        """
        if len(self.ds.targets) > 0:
            self.y=self.ds.df[self.ds.targets]
            self.X=self.ds.df.drop(self.ds.targets,axis=1)
        else:
            self.X=self.ds.df
            self.y=None
            
    def __get_ds(self):
        return self._ds

    def __set_ds(self,ds=None):
        if ds is None:
            raise ValueError("dataset should exist")
        else:
            self._ds = ds
            self.Xy()
           

    ds = property(__get_ds,__set_ds)
    
    def save_model(self,model_file=None):
        """export model to a file for later use
           model_file : set a specific model file; by default it is set to local 'flowka_model_*name.csv'
        """
        import pickle
        if model_file is None:
            model_file = './work/flowka_model_'+self.ds.name+'.dat'
        else:
            model_file=str(model_file).strip()
        try:
            columnorder=self.X.columns.tolist()#to keep the column order identical thru datasets 
            pickle.dump({'model':self._model,'scaler':self._scaler,'algo':self._algo, 'columnsorder':columnorder}, open(model_file, 'wb'))
            #add to dict : 'normalizer':self._normalizer
            print("model {0} saved".format(model_file))
        except ValueError:
            print("unable to save model {0}".format(model_file))

    def load_model(self,model_file=None):
        """export model to a file for later use
           model_file : set a specific model file; by default it is set to local 'flowka_model_*name.csv'
        """
        import pickle
        if model_file is None:
            model_file = './work/flowka_model_'+self.ds.name+'.dat'
        else:
            model_file=str(model_file).strip()
        try:
            load_dict = pickle.load(open(model_file, 'rb'))
            print("model {0} loaded".format(model_file))
        except:
            print("model {0} not found".format(model_file))
        
        try:
            self._model=load_dict['model']
            self._scaler=load_dict['scaler']
            self._algo=load_dict['algo']
            self.X=self.X[load_dict['columnsorder']]
            #self._normalizer=load_dict['normalizer']
        except:
                print("format {0} is invalid".format(model_file))

       
    def scale(self,scaler='std'):
        """standardize an normalize data
           standardize : center data around zero -> (Xi-mean(Xi))/StdDev(Xi)
           normalize : fit data around -1 and 1 -> (Xi/(max(Xi)-min(Xi))) 
           scaler : 'std', 'robust', uniform'44
        """
        if scaler == 'std':
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
        elif scaler == 'robust':
            from sklearn.preprocessing import RobustScaler
            self._scaler = robustScaler(quantile_range=(10, 90))
        else:
            from sklearn.preprocessing import QuantileTransformer
            self._scaler=QuantileTransformer(output_distribution='uniform')  
        #from sklearn.preprocessing import Normalizer
        #self._normalizer=Normalizer()
        columns=self.X.columns
        self.X=self._scaler.fit_transform(self.X)
        #self.X=self._normalizer.fit_transform(self.X)
        self.X=pd.DataFrame(self.X, columns=columns)
        return self.X

    def rescale(self):
        """re-execute the scaling
        """
        if self._scaler is None:
            return self.X
        columns=self.X.columns
        self.X=self._scaler.transform(self.X)
        #self.X=self._normalizer.fit_transform(self.X)
        self.X=pd.DataFrame(self.X, columns=columns)
        return self.X

    def unscale(self):
        """inverse scale transformation
        """
        columns=self.X.columns
        self.X=pd.DataFrame(self._scaler.inverse_transform(self.X), columns=columns)
        return self.X

    def linRegr(self):
        """Linear Regression
        """
        print('linRegr : Linear Regression')
        from sklearn.linear_model import LinearRegression
        self._model = LinearRegression()
        self.split(self.X, self.y)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)
  
    def logRegr(self):
        """Logistic Regression
        """
        print('logRegr : Logistic Regression')
        from sklearn.linear_model import LogisticRegression
        self._model=LogisticRegression()
        self.split(self.X, self.y)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)

    def polyfit(self,Xdata):
        """fit polynomial regression
        """
        from sklearn.preprocessing import PolynomialFeatures
        poly_reg=PolynomialFeatures(degree=4)
        Xpoly=poly_reg.fit_transform(Xdata)
        return Xpoly

    def polRegr(self):
        """Polynomial regression
        """
        print('polRegr : Polynomial Regression')
        from sklearn.linear_model import LinearRegression
        self._model=LinearRegression()
        self.split(self.polyfit(self.X), self.y)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)

    
    def knn(self,k=None):
        """k-Nearest Neighbors
        """
        print('knn : k-Nearest Neighbors')
        self.split(self.X, self.y)
        from sklearn.neighbors import KNeighborsClassifier
        if k is None:
            error_rate=[]
            preverr=[0,1]
            for k in range (1,30): #test over 30 values of k
                self._model=KNeighborsClassifier(n_neighbors=k)
                self.fit(self.X_train,self.y_train)
                self.predict(self.X_test)
                err=(np.mean(self._predictions != self.y_test))
                error_rate.append
                if err<preverr[1]: #if the error is lower than the lowest error stored
                    preverr=[k,err]#we store the new k and its error value
            print("k = {0} - error rate = {1}".format(str(preverr[0]),str(preverr[1])))
            k=preverr[0]
        self._model=KNeighborsClassifier(n_neighbors=k)#we choose the k value with the lowest error rate
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)#redo the prediction with chosen k
        
    def svm(self):
        """Support Vector Machine
        """
        print('svm : Support Vector Machine')
        self.split(self.X, self.y)
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        param_grid={'C':[0.1,0.5,1,5,10,50,100],'gamma':[1,0.5,0.1,0.05,0.01,0.005,0.001]}
        self._model = GridSearchCV(SVC(),param_grid,refit=True,verbose=0)#seearch the best parameters
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test) #fit and predict

    def cnn(self):
        print('cnn')
    
    def rnn(self):
        print('rnn')

    def ann(self):
        print('ann')
    
    def kmns(self,clusters=None):
        """K-Means
        """
        clusters=nvl(nvl(clusters,self.ds.clusters),2)#if no clsuters specified
        print('kmns : K-Means')
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        from random import shuffle
        import itertools as it
        lmlist=self.X.columns
        #if clusters is None:
        #     wcss=[]
        #     for clusters in range(1, self._kmns_max):
        #         #print ('{0} CLUSTERS '.format(str(clusters)))
        #         self._model = KMeans(n_clusters = clusters, init = 'k-means++', random_state = 0)
        #         self.fit(self.X)
        #         self.predict(self.X,fp=True)#fit_predict
        #         palette=shuffle(self.ds.fcol.flowkapalette)
        #         df=pd.concat([self.X,pd.DataFrame(data=self._predictions,columns=['kmn'])],axis=1)
        #         for lm in it.combinations(lmlist,2): 
        #             ax=sns.lmplot(x=lm[0],y=lm[1],data=df,hue='kmn',size=2,aspect=1,fit_reg=False,palette=palette,legend=True)
        #         plt.title('{0}} clusters'.format(str(clusters)), size=18)
        #         plt.show()
        #         wcss.append(self._model.inertia_)#elbow data
        #     #plot elbow    
        #     plt.plot(range(1, self._kmns_max), wcss,color=self.ds.fcol.color)
        #     plt.title('Elbow')
        #     plt.xlabel('clusters')
        #     plt.ylabel('WCSS')
        #     plt.show()
        #     #plot elbow
        #     #select a number of clusters
        #     clusters=int(input('type a cluster value :'))
        # #do it again with choosen clusters
        self._model = KMeans(n_clusters = clusters, init = 'k-means++', random_state = 0)
        self.fit(self.X)
        self.predict(self.X,fp=True)
        palette=self.ds.fcol.palette
        df=pd.concat([self.X,pd.DataFrame(data=self._predictions,columns=['kmn'])],axis=1)
        for lm in it.combinations(lmlist,2): 
            ax=sns.lmplot(x=lm[0],y=lm[1],data=df,hue=['kmn',None][clusters==1],size=3,aspect=1,fit_reg=False,palette=palette,legend=True)
        plt.title('{0} clusters for k-means'.format(str(clusters)), size=18)
        plt.show()
        #add predictions as a new column to the dataset
        self.ds.add=self.pdpred('kMeans')

    def spcl(self,clusters=None):
        """Spectral Clustering
        """
        clusters=nvl(nvl(clusters,self.ds.clusters),2)#if no clsuters specified
        print('spcl : Spectral Clustering')
        import matplotlib.pyplot as plt
        from sklearn.cluster import SpectralClustering
        from random import shuffle
        import itertools as it
        lmlist=self.X.columns
        #if clusters is None:
        #     wcss=[]
        #     for clusters in range(1, self._kmns_max):
        #         #print ('{0} CLUSTERS '.format(str(clusters)))
        #         self._model = SpectralClustering(n_clusters = clusters,, eigen_solver='arpack',affinity="nearest_neighbors")
        #         self.fit(self.X)
        #         self.predict(self.X)#predict
        #         palette=shuffle(self.ds.fcol.flowkapalette)
        #         df=pd.concat([self.X,pd.DataFrame(data=self._predictions,columns=['kmn'])],axis=1)
        #         for lm in it.combinations(lmlist,2): 
        #             ax=sns.lmplot(x=lm[0],y=lm[1],data=df,hue='kmn',size=2,aspect=1,fit_reg=False,palette=palette,legend=True)
        #         plt.title('{0}} clusters'.format(str(clusters)), size=18)
        #         plt.show()
        #         wcss.append(self._model.inertia_)#elbow data
        #     #plot elbow    
        #     plt.plot(range(1, self._kmns_max), wcss,color=self.ds.fcol.color)
        #     plt.title('Elbow')
        #     plt.xlabel('clusters')
        #     plt.ylabel('WCSS')
        #     plt.show()
        #     #plot elbow
        #     #select a number of clusters
        #     clusters=int(input('type a cluster value :'))
        # #do it again with choosen clusters
        self._model = SpectralClustering(n_clusters = clusters, eigen_solver='arpack',affinity="nearest_neighbors")
        self.fit(self.X)
        self.predict(self.X,fp=True)
        palette=self.ds.fcol.palette
        df=pd.concat([self.X,pd.DataFrame(data=self._predictions,columns=['spcl'])],axis=1)
        for lm in it.combinations(lmlist,2): 
            ax=sns.lmplot(x=lm[0],y=lm[1],data=df,hue=['spcl',None][clusters==1],size=3,aspect=1,fit_reg=False,palette=palette,legend=True)
        plt.title('{0} clusters for spectral clustering'.format(str(clusters)), size=18)
        plt.show()
        #add predictions as a new column to the dataset
        self.ds.add=self.pdpred('SpectralClustering')


    def afp(self):
        """Affinity Propagation
        """
        print('afp : Affinity Propagation')
        import matplotlib.pyplot as plt
        from sklearn.cluster import AffinityPropagation
        from random import shuffle
        import itertools as it
        lmlist=self.X.columns
        self._model = AffinityPropagation()
        self.fit(self.X)
        self._predictions=self._model.labels_
        clusters= np.shape(np.unique(self._predictions))[0]
        palette=self.ds.fcol.palette
        df=pd.concat([self.X,pd.DataFrame(data=self._predictions,columns=['afp'])],axis=1)
        for lm in it.combinations(lmlist,2): 
            ax=sns.lmplot(x=lm[0],y=lm[1],data=df,hue=['afp',None][clusters==1],size=3,aspect=1,fit_reg=False,palette=palette,legend=True)
        plt.title('{0} clusters for affinity propagation'.format(str(clusters)), size=18)
        plt.show()
        #add predictions as a new column to the dataset
        self.ds.add=self.pdpred('AffinityPropagation')

    def mns (self):
        """Mean Shift
        """
        print('mns : Mean Shift')
        import matplotlib.pyplot as plt
        from sklearn.cluster import MeanShift,estimate_bandwidth
        from random import shuffle
        import itertools as it
        lmlist=self.X.columns
        self._model = MeanShift()
        self.fit(self.X)
        self._predictions=self._model.labels_
        clusters= np.shape(np.unique(self._predictions))[0]
        palette=self.ds.fcol.palette
        df=pd.concat([self.X,pd.DataFrame(data=self._predictions,columns=['mns'])],axis=1)
        
        for lm in it.combinations(lmlist,2): 
            ax=sns.lmplot(x=lm[0],y=lm[1],data=df,hue=['mns',None][clusters==1],size=3,aspect=1,fit_reg=False,palette=palette,legend=True)
        plt.title('{0} clusters for mean shift'.format(str(clusters)), size=18)
        plt.show()
        #add predictions as a new column to the dataset
        self.ds.add=self.pdpred('meanShift')

    def dbs (self):
        """DBScan
        """
        print('dbs : db Scan')
        import matplotlib.pyplot as plt
        from sklearn.cluster import DBSCAN
        from random import shuffle
        import itertools as it
        lmlist=self.X.columns
        self._model = DBSCAN()
        self.fit(self.X)
        self._predictions=self._model.labels_
        clusters= np.shape(np.unique(self._predictions))[0]
        palette=self.ds.fcol.palette
        df=pd.concat([self.X,pd.DataFrame(data=self._predictions,columns=['dbs'])],axis=1)
        for lm in it.combinations(lmlist,2): 
            ax=sns.lmplot(x=lm[0],y=lm[1],data=df,hue=['dbs',None][clusters==1],size=3,aspect=1,fit_reg=False,palette=palette,legend=True)
        plt.title('{0} clusters for dbScan'.format(str(clusters)), size=18)
        plt.show()
        #add predictions as a new column to the dataset
        self.ds.add=self.pdpred('dbScan')


    def forClass(self):
        """Random Forest Classifier'
        """
        print('forClass : Random Forest Classifier')
        self.split(self.X, self.y)
        from sklearn.ensemble import RandomForestClassifier
        self._model=RandomForestClassifier(n_estimators=600)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)


    def treeClass(self):
        """Decision Tree Classifier
        """
        print('treeClass : Decision Tree Classifier')
        self.split(self.X, self.y)
        from sklearn.tree import DecisionTreeClassifier
        self._model=DecisionTreeClassifier()
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)

    def forRegr(self):
        """Random Forest Regressor'
        """
        print('forRegr : Random Forest Regressor')
        self.split(self.X, self.y)
        from sklearn.ensemble import RandomForestRegressor
        self._model=RandomForestRegressor(n_estimators=600)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)

    def treeRegr(self):
        """Decision Tree Regressor
        """
        print('treeRegr : Decision Tree Regressor')
        self.split(self.X, self.y)
        from sklearn.tree import DecisionTreeRegressor
        self._model=DecisionTreeRegressor()
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)

    def predict(self,inputs=None,fp=False):
        """predict results from inputs
           fp: execute 'fit_predict' instead of 'predict'
        """
        inputs=[inputs,self.X][inputs is None]

        if fp==False:
            self._predictions=self._model.predict(inputs)
        else:
            self._predictions=self._model.fit_predict(inputs)

        return  self._predictions

    def fit(self,X,y=None):
        """fit model over input and output training data
        """    
        if y is None:
            self._model.fit(X)
        else:
            self._model.fit(X,y)

    def split(self,X,y):
        """split dataset in training and testing data
        """  
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=self.test_size, random_state=101)

    def pdpred(self,name=None):
        """return predictions into a dataframe
        """
        return pd.DataFrame(self._predictions,columns=[nvl(name,'predictions')])

    def __get_algo(self):
        return self._algo
    
    def __set_algo(self,algo='linRegr'):
        """set the algorithm and train the model
            algo in ['linRegr' :'linear regression','logRegr':'logistic regression',
            'polRegr':polynomial regression', 'knn':'k-nearest neighbors','ann':'artificial neural network',
            'cnn':'convolutional neural network,'rnn':' recurrent neural network',
            'kmns':'k-means', 'forRegr':'random forest regressor','treeRegr':'decision tree regressor',
            'forClass':'random forest classifier','treeClass':'decision tree classifier'
            'svm':'support vector machine', 'mns':'mean shift', 'dbs':'dbscan','afp':'afinProp','spcl':'spClust'}
        """
        algodict={'linRegr':self.linRegr,'logRegr':self.logRegr,'polRegr':self.polRegr,
                  'ann':self.ann,'knn':self.knn,'cnn':self.cnn,'rnn':self.rnn,
                  'kMeans':self.kmns,'forRegr':self.forRegr,'treeRegr':self.treeRegr,
                  'forClass':self.forClass,'treeClass':self.treeClass,
                  'svm':self.svm, 'meanShift':self.mns, 'dbScan':self.dbs,'afinProp':self.afp,'spClust':self.spcl}
        if not algo in algodict:
            raise ValueError("algo does not exist")
        self._algo = algo
        self.ds.fmd.algo=self._algo
        algodict[algo]()    
        
    algo = property(__get_algo,__set_algo)

    def __get_lrcoef(self):
        """give the coefficients of a linear regression
        """
        self._lrcoef=pd.DataFrame(self._model.coef_,self.ds.df.drop(self.ds.targets,axis=1).columns)
        self._lrcoef.columns = ['Coeffecient']
        return self._lrcoef
    
    lrcoef =property(__get_lrcoef)
    
    def predictplot(self):
        """Create a scatter plot of the real test values versus the predicted values.
           if targets is not binary nor classification : scatterplot
           else confusion matrix
        """ 
        if self.ds.details.loc['distinct',self.ds.targets]>2 and str(self.ds.details.loc['dtype',self.ds.targets])!='category':
            sns.color_palette=self.ds.fcol.palette
            sns.edgecolor=self.ds.fcol.c('flowkadark')

            sns.regplot(x=self.y_test, y=self._predictions, color=self.ds.fcol.color)
            
    def metrics(self):
        """evaluate the model performance by calculating 
           the residual sum of squares and the explained variance score (R^2).
           Mean Absolute Error - Mean Squared Error - Root Mean Squared Error
           Confusion Matrix :
                          Predicted True / Predicted False
           observed True      TN               FP
           observed False     FN               TN
           
           PRECISION : (True) TP/(FP+TP) *** (False) TN/(FN+TN)
           RECALL : (True) TP/(TP+FN) (sensitivity) *** (False) TN/(TN+FP) (specificity)
           F1-SCORE : 2xPRECISIONxRECALL/(PRECISION+RECALL)
           ACCURACY : (TP+TN)/(TP+TN+FP+FN)
           ERROR RATE = (FP+FN)/(TP+TN+FP+FN)
        """
        from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, silhouette_score
        self._metrics={'MAE':None,'MSE':None,'RMSE':None,'CM':None,'AR':None,'ER':None,'SC':None,'Error':None} 
        if self._algo in ['logRegr','knn','forClass','treeClass','svm']:
            cm=confusion_matrix(self.y_test,self._predictions,labels=list(set(self.y_test)))
            ar=cm.trace()/cm.sum()# accuracy rate -> sum(diag)/sum()
            er=(cm.sum()-cm.trace())/cm.sum()#error rate -> (sum()-sum(diag))/sum()
            self._metrics.update({'CM':cm,'AR':ar,'ER':er,'Error':er}) 
        elif self._algo in ['linRegr','forRegr','treeRegr','polRegr']:
            mae=mean_absolute_error(self.y_test, self._predictions)
            mse=mean_squared_error(self.y_test, self._predictions)
            rmse=np.sqrt(mse)
            self._metrics.update({'MAE':mae,'MSE':mse,'RMSE':rmse,'Error':rmse})
        elif self.algo in ['kMeans','spClust','meanShift','dbScan','afinProp']: 
            if np.shape(np.unique(self._predictions))[0]<2:
                sc=99999
            else:
                sc=silhouette_score(self.X, self._predictions) 
            self._metrics.update({'Error':sc,'SC':sc})
        else:    
            none=None
        return self._metrics

    def reports(self):
        """print the Metrics
        """
        self.metrics()
        
        if self._algo in ['logRegr','knn','forClass','treeClass','svm']:
            from sklearn.metrics import classification_report
            print(pd.DataFrame(data=self._metrics['CM'],columns=list(set(self.y_test)),index=list(set(self.y_test))))
            print("\n")
            print(classification_report(self.y_test,self._predictions))
            print("\n")
            print ('accuracy rate \t: {0}\nerror rate \t: {1}'.format(self._metrics['AR'],self._metrics['ER'])) 
            print("\n")
            sns.heatmap(self._metrics['CM'],cmap=self.ds.fcol.cmap,annot=True,fmt='.0f')
            plt.xlabel('Prediction')
            plt.ylabel('Observation')
        elif self._algo in ['linRegr','forRegr','treeRegr','polRegr']:
            print ('Mean Absolut Error (MAE) : {0}\nMean Squared Error (MSE) : {1}\nRoot Mean Squared Error (RMSE) : {2}'.format(self._metrics['MAE'],self._metrics['MSE'],self._metrics['RMSE']) )
            self.predictplot()
            if self._algo in ['linRegr']:
                print(self.lrcoef) #coef de regression
                #residual plot
                sns.distplot(self.y_test-self._predictions,bins=50,kde=False,color=self.ds.fcol.color)
        elif  self.algo in ['kMeans','spClust','meanShift','dbScan','afinProp']: 
                print ('Silhouette Score (SC) : {0}\n'.format(self._metrics['SC']))
        plt.show()

    def detect_algo(self):
        """detect the ml algo based on targets category and values
        """
        algos=['logRegr','linRegr','polRegr','forRegr','treeRegr',
        'forClass','treeClass','svm','ann','knn','cnn','rnn',
        'kMeans','spClust',
        'meanShift','dbScan','afinProp'] 
        algolist=[]
        if len(self.ds.targets)==0 and self.ds.clusters is not None:
                algolist.extend(algos[12:14]) #clustering cluster is known
        elif len(self.ds.targets)==0 and self.ds.clusters is None:
                algolist.extend(algos[14:17]) #clustering cluster is not known
        elif (str(self.ds.details.loc['dtype',self.ds.targets]).find('int')>-1 and self.ds.details.loc['distratio',self.ds.targets]<0.1) or (str(self._ds.details.loc['dtype',self._ds.targets])=='category'):   
                algolist.extend(algos[5:8])#classification
                if (self.ds.details.loc['max',self.ds.targets]==1) and (self.ds.details.loc['distinct',self.ds.targets]==2):
                    algolist.extend(algos[0:1])#logistic regression
                if (self.ds.details.loc['count',self.ds.targets]<self._knn_max) and (self.ds.details.loc['distinct',self.ds.targets]<10):
                    algolist.extend(algos[9:10])#knn classification
        elif isinlist(self.ds.targets,self.ds.dtypes['continuous']):
                algolist.extend(algos[1:5])#regression
        print("algos : {0}".format(algolist))
        return algolist

    def best_predict(self,algolist):
        """ test all available algorithm for a data problem and keep the best (less Mean Square Error)
        """
        bestpredict=[]
        if algolist is None:
            print("No algorithms to test")
            return -1
        elif len(algolist)>1:
            for testalgo in algolist:
                print(testalgo)
                self.algo=testalgo
                Error=self.metrics()['Error']
                print("Error:{0}".format(Error))
                if len(bestpredict)==0 or Error<bestpredict[1]:
                    bestpredict=[testalgo,Error]
        else :
            bestpredict.extend(algolist)
        print("choose : {0}\n".format(bestpredict[0]))
        self.algo=bestpredict[0]# run one more time the best algo 
        self.ds.fmd.algo=self.algo














