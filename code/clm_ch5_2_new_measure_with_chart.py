
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
               quantifiers= {'Base':1000, 'aad':0.4},
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

clm.print_rulelist()
clm.print_summary()


#Prepare a new list of rules that satisfy the new measure of interestingness
# - loop over rule list, for each
#   - get fourfold table via API
#   - calculate own measure of interestingness (combines support and confidence)
#   - if interestingness is over a threshold, append to a new list

d = []
dg = []
ordr = 0
for i in range(clm.get_rulecount()):
    ff = clm.get_fourfold(i+1)
    conf = None
    if ff[0]+ff[1]>0:
        conf = ff[0]/(ff[0]+ff[1])
    base = ff[0]
    interestingness = base/500 + 5*math.exp(50*(conf-0.019))
    dg_record = {
            'Rule_id': i + 1,
            'Base': base,
            'Order': "",
            'Confidence': conf,
            'Hue': '2',
            'Interestingness': interestingness,
            'Text': clm.get_ruletext(i + 1)
        }
    if interestingness > 19: #ignore this condition to get chart for all data [e.g., use interestingness >0]
        ordr += 1
        d.append(
            {
                'Rule_id': i + 1,
                'Order': ordr,
                'Base': base,
                'Confidence': conf,
                'Interestingness': interestingness,
                'Text': clm.get_ruletext(i + 1)
            })
        dg_record['Hue']='6'
        dg_record['Order'] = ordr
    dg.append(dg_record)

#draw a scatterplot with rules from a new list
df_r = pd.DataFrame(dg)
ax=sns.scatterplot(data=df_r,x="Base",y="Confidence",hue="Hue",hue_order=['0','1','2','3','4','5','6'],palette="Blues",legend=False)
for line in range(0, df_r.shape[0]):
    ax.text(df_r["Base"][line] + 0.01, df_r["Confidence"][line],
            str(df_r["Order"][line]), horizontalalignment='left',
            size='medium', color='black', weight='semibold')

#print all rules in a new list

print(f"#   OrigId  Score  Base  Conf    Rule")
for i in range(ordr):
    print(f"{i+1:>3d}    {d[i]['Rule_id']:>3d}  {d[i]['Interestingness']:.2f}  {d[i]['Base']}  {d[i]['Confidence']:.4f} {d[i]['Text']}")

plt.show()


