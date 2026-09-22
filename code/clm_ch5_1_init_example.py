
import pandas as pd
from sklearn.impute import SimpleImputer
from cleverminer import cleverminer
from matplotlib import pyplot as plt
import seaborn as sns
import math

df = pd.read_csv ('./data/accidents.zip', encoding='cp1250', sep='\t')
df=df[['Driver_Age_Band','Driver_IMD','Sex','Area','Journey','Road_Type','Speed_limit','Light','Vehicle_Location','Vehicle_Type','Vehicle_Age','Hit_Objects_in','Hit_Objects_off','Casualties','Severity']]

#fill in missing values

imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

#run the CleverMiner task

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'Base':2000, 'aad':1.0},
               ante ={
                    'attributes':[
                        {'name': 'Driver_Age_Band', 'type': 'seq', 'minlen': 1, 'maxlen': 3},
                        {'name': 'Vehicle_Type', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Sex', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Speed_limit', 'type': 'seq', 'minlen': 1, 'maxlen': 3},
                        {'name': 'Journey', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':4, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'Severity', 'type': 'lcut', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1 , 'type':'con'}
                  ,opts = {'keep_df':True}
                  )

#check how CleverMiner ordered categories
clm.print_data_definition()

clm.print_rulelist()
clm.print_summary()

clm.print_rule(1)
clm.draw_rule(1)
