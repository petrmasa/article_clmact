
import pandas as pd
import sys
import sklearn.impute
from matplotlib import pyplot as plt

from sklearn.impute import SimpleImputer

from cleverminer import cleverminer

df = pd.read_csv ('./data/accidents.zip', encoding='cp1250', sep='\t')
df=df[['Driver_Age_Band','Driver_IMD','Sex','Area','Journey','Road_Type','Speed_limit','Light','Vehicle_Location','Vehicle_Type','Vehicle_Age','Hit_Objects_in','Hit_Objects_off','Casualties','Severity']]

#df=df[['Driver_Age_Band','Driver_IMD','Sex','Journey','Hit_Objects_in','Hit_Objects_off','Casualties','Severity']]

imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)


clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'Base':20000, 'aad':1.0},
               ante ={
                    'attributes':[
                        {'name': 'Driver_Age_Band', 'type': 'seq', 'minlen': 1, 'maxlen': 3},
                        {'name': 'Driver_IMD', 'type': 'seq', 'minlen': 1, 'maxlen': 3},
                        {'name': 'Sex', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Journey', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Area', 'type': 'subset', 'minlen': 1, 'maxlen': 2},
                        {'name': 'Road_Type', 'type': 'subset', 'minlen': 1, 'maxlen': 2},
                        {'name': 'Speed_limit', 'type': 'seq', 'minlen': 1, 'maxlen': 2},
                        {'name': 'Light', 'type': 'subset', 'minlen': 1, 'maxlen': 2},
                        {'name': 'Vehicle_Location', 'type': 'subset', 'minlen': 1, 'maxlen': 2},
                        {'name': 'Vehicle_Type', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Vehicle_Age', 'type': 'seq', 'minlen': 1, 'maxlen': 11}
                    ], 'minlen':1, 'maxlen':3, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'Casualties', 'type': 'rcut', 'minlen': 1, 'maxlen': 6},
                        {'name': 'Severity', 'type': 'lcut', 'minlen': 1, 'maxlen': 2}
                    ], 'minlen':1, 'maxlen':1 , 'type':'con'}
               )

clm.print_rulelist()
clm.print_summary()
