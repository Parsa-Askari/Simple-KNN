import numpy as np
import pandas as pd
from tqdm.auto import tqdm
class KNN: #
    def __init__(self,X,Y):
        if(type(X)!=pd.core.frame.DataFrame): #  change data type to pandas dataframe
            self.Features=pd.DataFrame(X).copy()
        if(type(Y)!=pd.core.frame.DataFrame): 
            self.Labels=pd.DataFrame(Y,columns=["Y"]).copy()
        self.Database=pd.concat([self.Features,self.Labels],axis=1) # create the initial database with labels and features
    def predict(self,X,K):
        n,_=X.shape
        Classes=[]
        for i in tqdm(range(n)):
            dist=np.linalg.norm(X[i]-self.Features.to_numpy(),axis=1) # calculate the distance from the target with L2 norm
            
            
            self.Database["dist"]=dist # save distance in database
            self.Database=self.Database.sort_values(by="dist") # sort database based of the distance
            
            Class=self.Database["Y"].head(K).value_counts(ascending=False).index[0] # find the most repeated class
            Classes.append(Class)

            self.Database=self.Database.sort_index() # return the database to initial form
        return np.array(Classes)