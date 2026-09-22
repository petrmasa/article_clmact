
import pandas as pd

from sklearn.impute import SimpleImputer

from cleverminer.actplusminer import actplusminer
from cleverminer import cleverminer


df = pd.read_csv ('./data/accidents.zip', encoding='cp1250', sep='\t')
df=df[['Driver_Age_Band','Vehicle_Type','Sex','Speed_limit','Journey','Severity']]

#fill-in missing values
imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

#run the CleverMiner task to find rules
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

clm.print_data_definition()

#print rules and print the summary
clm.print_rulelist()
clm.print_summary()


#for us, rule #1 is interesting (too much Fatal accidents). We will print it completely
clm.print_rule(1)
clm.draw_rule(1)

#now, we will find rules similar to rule #1 where Sex and Driver_Age_Band can be different to find subgroups of drivers where fatality is <=50% of fatality in rule #1
apm = actplusminer(clm=clm,rule_id=1,quantifiers={'Base':100, 'confratio_leq': 0.5},ante_flex=[{'name': 'Sex', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Driver_Age_Band', 'type': 'seq', 'minlen': 1, 'maxlen': 3}],succ_flex=None)

#print of the results
apm.print_rulelist()
apm.draw_rule(3)

#for sure, we may check how CleverMiner undestands the dataset and if it ordered categories in a proper way
clm.print_data_definition()
