import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.loda import LODA


from statsmodels.discrete.discrete_model import Probit, Logit, \
    GeneralizedPoisson, Poisson
import matplotlib.pyplot as plt

"""
Attend
# Estimate a linear regression relating atndrte to ACT, priGPA, frosh, and soph; 
compute the usual OLS standard errors. Interpret the coe‰cients on ACT and priGPA. 
Are any fitted values outside the unit interval?

http://fmwww.bc.edu/ec-p/data/wooldridge/attend.des

Smoke
# Use a linear regression model to explain cigs, the number of cigarettes smoked
per day. Use as explanatory variables logðcigpricÞ, logðincomeÞ, restaurn, white,
educ, age, and age2. Are the price and income variables significant? Does using
heteroskedasticity-robust standard errors change your conclusions?

http://fmwww.bc.edu/ec-p/data/wooldridge/smoke.des


BWGHT
# Define a binary variable, smikes, if the women smokes during the pregnancy
estimate a probit model relating smokes to mothereduc, white, and log(faminc)
http://fmwww.bc.edu/ec-p/data/wooldridge/bwght.des


Grogger
Define a binary variable, say arr86, equal to unity if a man was arrested at least
once during 1986, and zero otherwise. Estimate an LPM relating arr86 to pcnv, avgsen,
tottime, ptime86, inc86, black, hispan, and born60. Report the usual and heteroskedasticity-
robust standard errors. 
http://fmwww.bc.edu/ec-p/data/wooldridge2k/GROGGER.DES

"""

dta_lists = [
    "attend.dta",  
    # "smoke.dta",
    # "bwght.dta",
    # "grogger.dta",
]

# OD_Flag = False
OD_Flag = True

for i in range(len(dta_lists)):
    df = pd.read_stata(os.path.join("data", dta_lists[i]))
    print(df.columns)


    # attend    
    if dta_lists[i] == "attend.dta":
        Y = df['atndrte']
        X = df[['priGPA', 'ACT', 'frosh', 'soph']]
    
    elif dta_lists[i] == "smoke.dta":
        # smoke
        Y = df['cigs']
        X = df[['educ', 'cigpric', 'white', 'age', 'income', 'restaurn',
            'lincome', 'agesq', 'lcigpric']] 
        X = X.fillna(X.mean())
    
    elif dta_lists[i] == "bwght.dta":
        # bwght
        df['smokes'] = (df['cigs'] >= 1).astype(int)
        Y = df['smokes']
        X = df[['motheduc', 'white', 'lfaminc']] 
        X = X.fillna(X.mean())
    else:
        # grogger
        df['arr86'] = (df['narr86']>=1).astype(int)
        Y = df['arr86']
        X = df[['pcnv', 'avgsen', 'tottime', 'ptime86', 'inc86', 'black', 'hispan', 'born60']]
    
    print(i, X.shape, Y.shape)
    
    if OD_Flag:
        
        # clf = HBOS(contamination=0.05)
        # clf = IForest(contamination=0.05)
        clf = LODA(contamination=0.05)
        clf.fit(X)
        
        # remove outliers
        X = X.loc[np.where(clf.labels_==0)]
        Y = Y.loc[np.where(clf.labels_==0)]
    
    X = sm.add_constant(X)

    # general OLS
    # https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
    # model=sm.OLS(Y, X.astype(float))

    # robust regression 
    # https://www.statsmodels.org/stable/generated/statsmodels.robust.robust_linear_model.RLM.html
    model=sm.RLM(Y, X.astype(float))

    # probit model 
    # https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Probit.html
    # model = Probit(Y, X.astype(float))

    # logit model 
    # https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html
    # model = Logit(Y, X.astype(float))

    # poisson model 
    # https://www.statsmodels.org/stable/generated/statsmodels.formula.api.poisson.html
    # model = Poisson(Y, X.astype(float))

    final_model = model.fit()
    results_summary = final_model.summary()
    print(results_summary)
    results_as_html = results_summary.tables[1].as_html()
    result_df = pd.read_html(results_as_html, header=0, index_col=0)[0]

    print(result_df.to_latex())
