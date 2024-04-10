import numpy as np
import pandas as pd
from tqdm.auto import tqdm
class KNN: #
    def __init__(self,X,Y):
        if(type(X)!=pd.core.frame.DataFrame):
            self.Features=pd.DataFrame(X).copy()
        if(type(Y)!=pd.core.frame.DataFrame):
            self.Labels=pd.DataFrame(Y,columns=["Y"]).copy()
        self.Database=pd.concat([self.Features,self.Labels],axis=1)
    def predict(self,X,K):
        n,_=X.shape
        Classes=[]
        for i in tqdm(range(n)):
            dist=np.linalg.norm(X[i]-self.Features.to_numpy(),axis=1)
            
            
            self.Database["dist"]=dist
            self.Database=self.Database.sort_values(by="dist")
            
            Class=self.Database["Y"].head(K).value_counts(ascending=False).index[0]
            Classes.append(Class)

            self.Database=self.Database.sort_index()
        return np.array(Classes)