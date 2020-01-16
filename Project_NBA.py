#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import itertools
import time
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn import datasets, linear_model, tree, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression, make_hastie_10_2
from scipy.cluster.vq import vq, kmeans, kmeans2, whiten
get_ipython().run_line_magic('matplotlib', 'inline')
os.chdir(r"U:\RemoteApp")

stats = pd.read_excel(r"nba-players-stats\Seasons_Stats.xlsx") #mother data set for performance stats
stats = stats[stats['Tm']!='TOT'] #remove players with team name TOT. These entries were traded during the regular season.
salary17 = pd.read_csv(r"salary\salary.csv") #data set for 2017 salary
aggr = {'Pos':'first','Tm':'last','G':'sum','GS':'sum','MP':'sum','PER':'mean','TS%':'mean','3PAr':'mean','FTr':'mean',
        'ORB%':'mean','DRB%':'mean','TRB%':'mean','AST%':'mean','STL%':'mean','BLK%':'mean','TOV%':'mean','USG%':'mean',
        'OWS':'mean','DWS':'mean','WS':'mean','OBPM':'mean','DBPM':'mean','BPM':'mean','VORP':'mean','FG':'sum','FGA':'sum',
        'FG%':'mean','3P':'sum','3P%':'mean','2P':'sum','2P%':'mean','eFG%':'mean','FT':'sum','FT%':'mean','ORB':'sum',
        'DRB':'sum','TRB':'sum','AST':'sum','STL':'sum','BLK':'sum','TOV':'sum','PF':'sum','PTS':'sum'}

#segregate and group by player names for each season from 2013-16 to generate unique data points
stats13 = stats[stats["Year"] == 2013.0]
stats13 = stats13.groupby(["Player"], as_index=False).agg(aggr)
stats14 = stats[stats["Year"] == 2014.0]
stats14 = stats14.groupby(["Player"], as_index=False).agg(aggr)
stats15 = stats[stats["Year"] == 2015.0]
stats15 = stats15.groupby(["Player"], as_index=False).agg(aggr)
stats16 = stats[stats["Year"] == 2016.0]
stats16 = stats16.groupby(["Player"], as_index=False).agg(aggr)
stats17 = stats[stats["Year"] == 2017.0]
stats17 = stats17.groupby(["Player"], as_index=False).agg(aggr)

#consolidate 2013-16 data
stats1316 = pd.concat([stats13,stats14,stats15,stats16], ignore_index=True)
salary17 = salary17.groupby(["Player"]).agg({'Tm':'last', 'season17_18':'last'})
print (salary17.head(5))

#merge salary with performance stats and clean the data
stats17 = pd.merge(stats17, salary17, how='left', on='Player')
stats17 = stats17.drop(['Tm_y'], axis=1); stats17 = stats17.drop(['PER'], axis=1)
stats17 = stats17.rename(columns={'Tm_x':'Tm', 'season17_18':'SALARY'})

#segregate data for Indiana Pacers
pacers = stats17[stats17.Tm == "IND"]
stats17 = stats17[stats17.Tm != "IND"]
cols = ['TS%','TRB%','AST%','TOV%','USG%','BPM','PF']


# In[ ]:


plt.style.use('ggplot')
#from tqdm import tnrange, tqdm_notebook
def fit_linear_reg(X,Y):
    #Fit ridge linear regression model and return RSS and R squared values
    model_k = linear_model.Ridge(alpha=.5)
    model_k.fit(X,Y)
    RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X,Y)
    return RSS, R_squared
    
#Initialization variables
Y = stats1316.PER
X = stats1316[['G','MP','TS%','TRB%','AST%','STL%','BLK%','TOV%','USG%','WS','BPM','VORP','FG%','3P%','FT%','PF','PTS']]
k = 17
X = X.fillna(0)
Y = Y.fillna(0)
RSS_list, R_squared_list, feature_list = [],[], []
numb_features = []

#Looping over k = 1 to k = 17 features in X
#for k in tnrange(1,len(X.columns) + 1, desc = 'Loop...'):
for k in range(1,len(X.columns) + 1):
    #Looping over all possible combinations: from 17 choose k
    for combo in itertools.combinations(X.columns,k):
        tmp_result = fit_linear_reg(X[list(combo)],Y)#Store temp result 
        RSS_list.append(tmp_result[0])#Append lists
        R_squared_list.append(tmp_result[1])
        feature_list.append(combo)
        numb_features.append(len(combo))   

#Store in DataFrame
df = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, 'R_squared':R_squared_list,'features':feature_list})

#Print the best features for no. of features
df_min = df[df.groupby('numb_features')['RSS'].transform(min) == df['RSS']]
df_max = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]
display(df_min.head(8))
display(df_max.head(8))


# In[26]:


#clean empty cells in dataframes
stats1316 = stats1316.fillna(0)
stats17 = stats17.fillna(0)

#choose parameters and predictor for regression
perf_var1316 = stats1316[cols] #training params
perf_var17 = stats17[cols] #test params
per = stats1316.PER

#select best regression model
#regr1 = linear_model.LinearRegression()
regr1 = linear_model.Ridge(alpha=0.5)
#regr1 = tree.DecisionTreeRegressor()
#regr1 = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
#regr1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='huber')

regr1.fit(perf_var1316, per) #fit the model based on training data
per_pred = regr1.predict(perf_var17) #predict PER values for 2017
stats17["PER"] = per_pred; stats17 = stats17.sort_values(by=['PER'], ascending=False)

#perform k-means clustering to categorize players based on performance and salary
whitened = whiten(stats17[['PER','SALARY']].values) #normalize data
centroids,_ = kmeans(whitened, 2)
idx,_ = vq(whitened,centroids)
#generate clustering scatterplot
plt.plot(whitened[idx==0,0], whitened[idx==0,1], 'ob',
         whitened[idx==1,0], whitened[idx==1,1],'og')
plt.plot(centroids[:,1], centroids[:,1], 'sm', c='r', markersize='8')
plt.figure(figsize=(10, 10))
plt.show()

print("Score: "+str(regr1.score(perf_var1316, per)))
print(stats17[['Player','PER']])


# In[27]:


#predict top 5 players in Indiana Pacers
pacer_data = pacers[cols]; print(pacer_data)
pacers_pie = regr1.predict(pacer_data); pacers["PER"] = pacers_pie
pacers = pacers.sort_values(by=['PER'], ascending=False); print(pacers[['Player','PER']])
index = np.arange(len(pacers))
plt.bar(index, pacers['PER'], color=['g','g','g','g','g','b','b','b','b','b','b','b','b'])
plt.xlabel('Players', fontsize=11)
plt.ylabel('PER Score', fontsize=11)
plt.xticks(index, pacers['Player'], fontsize=9, rotation=30)
plt.title('Indiana Pacers - Player Ratings')
plt.figure(figsize=(12, 12))


# In[28]:


top5 = pacers[:5] #segregate top 5 players in IND Pacers

#calculate available salary cap for choosing remaining roster
salary_cap = 120000000
avail_sal = salary_cap - top5.iloc[:5,-2].sum()

#calculate position based stats for top 5
pos = top5.groupby(["Pos"], as_index=False).agg({'SALARY':'sum'}); temp = top5.Pos.value_counts().to_frame(); temp=temp.sort_values(by=['Pos'], ascending=True).reset_index()
pos["Count"] = temp['Pos']

#create position based dataframes for rest of NBA
pg = stats17[stats17['Pos']=='PG'].sort_values(by=['PER'], ascending=False)
sg = stats17[stats17['Pos']=='SG'].sort_values(by=['PER'], ascending=False)
pf = stats17[stats17['Pos']=='PF'].sort_values(by=['PER'], ascending=False)
sf = stats17[stats17['Pos']=='SF'].sort_values(by=['PER'], ascending=False)
c = stats17[stats17['Pos']=='C'].sort_values(by=['PER'], ascending=False)

#create position based dataframes for IND Pacers
ind_pg = pacers[pacers['Pos']=='PG'].sort_values(by=['PER'], ascending=False)
ind_sg = pacers[pacers['Pos']=='SG'].sort_values(by=['PER'], ascending=False)
ind_pf = pacers[pacers['Pos']=='PF'].sort_values(by=['PER'], ascending=False)
ind_sf = pacers[pacers['Pos']=='SF'].sort_values(by=['PER'], ascending=False)
ind_c = pacers[pacers['Pos']=='C'].sort_values(by=['PER'], ascending=False)

#create salary and composition stats for rest of NBA
nba_sg = [sg.SALARY.count(), sg.SALARY.mean(), sg.SALARY.count()/(stats17.Tm.nunique()+1)]
nba_sf = [sf.SALARY.count(), sf.SALARY.mean(), sf.SALARY.count()/(stats17.Tm.nunique()+1)]
nba_pg = [pg.SALARY.count(), pg.SALARY.mean(), pg.SALARY.count()/(stats17.Tm.nunique()+1)]
nba_c = [c.SALARY.count(), c.SALARY.mean(), c.SALARY.count()/(stats17.Tm.nunique()+1)]
nba_pf = [pf.SALARY.count(), pf.SALARY.mean(), pf.SALARY.count()/(stats17.Tm.nunique()+1)]
nba_sal = [stats17.SALARY.count(), stats17.SALARY.mean(), stats17.SALARY.count()/(stats17.Tm.nunique()+1)]

#create budget allocation parameters(average budget, benchmarked budget) for each position in IND Pacers
sg_budget = [ind_sg.SALARY.count(), ind_sg.SALARY.mean(), (ind_sg.SALARY.count()*nba_sg[1]*nba_sal[2])/pacers.SALARY.count()]
sf_budget = [ind_sf.SALARY.count(), ind_sf.SALARY.mean(), (ind_sf.SALARY.count()*nba_sf[1]*nba_sal[2])/pacers.SALARY.count()]
pg_budget = [ind_pg.SALARY.count(), ind_pg.SALARY.mean(), (ind_pg.SALARY.count()*nba_pg[1]*nba_sal[2])/pacers.SALARY.count()]
c_budget = [ind_c.SALARY.count(), ind_c.SALARY.mean(), (ind_c.SALARY.count()*nba_c[1]*nba_sal[2])/pacers.SALARY.count()]
pf_budget = [ind_pf.SALARY.count(), ind_pf.SALARY.mean(), (ind_pf.SALARY.count()*nba_pf[1]*nba_sal[2])/pacers.SALARY.count()]
pac_budget = [pacers.SALARY.count(), pacers.SALARY.mean(), nba_sal[1]*nba_sal[2]]

#calculate position-wise available budget for IND Pacers
c_budget.append((c_budget[2]*salary_cap)/pac_budget[2])
sg_budget.append((sg_budget[2]*salary_cap)/pac_budget[2])
sf_budget.append((sf_budget[2]*salary_cap)/pac_budget[2])
pg_budget.append(((pg_budget[2]*salary_cap)/pac_budget[2])+c_budget[3])
pf_budget.append((pf_budget[2]*salary_cap)/pac_budget[2])
pac_budget.append(salary_cap)

#generate PG recommendations based on budget allocation and PER score
final_pg = pg[pg['SALARY'] < (pg_budget[3]-top5.iloc[2,-2])]; final_pg = final_pg[["Player","SALARY","PER"]]
final_pg = final_pg[:10]; final_pg = final_pg.sort_values(by="SALARY",ascending=True); final_pg = final_pg.reset_index(); final_pg = final_pg.drop(['index'], axis=1)

#generate SG recommendations based on budget allocation and PER score
final_sg = sg[sg['SALARY'] < sg_budget[3]]; final_sg = final_sg[["Player","SALARY","PER"]]
final_sg = final_sg[:10]; final_sg = final_sg.sort_values(by="SALARY",ascending=True); final_sg = final_sg.reset_index(); final_sg = final_sg.drop(['index'], axis=1)

#generate SF recommendations based on budget allocation and PER score
final_sf = sf[sf['SALARY'] < (sf_budget[3]-top5.iloc[0,-2])]; final_sf = final_sf[["Player","SALARY","PER"]]
final_sf = final_sf[:10]; final_sf = final_sf.sort_values(by="SALARY",ascending=True); final_sf = final_sf.reset_index(); final_sf = final_sf.drop(['index'], axis=1)

#generate PF recommendations based on budget allocation and PER score
final_pf = pf[pf['SALARY'] < (pf_budget[3]-top5.iloc[4,-2])]; final_pf = final_pf[["Player","SALARY","PER"]]
final_pf = final_pf[:10]; final_pf = final_pf.sort_values(by="SALARY",ascending=True); final_pf = final_pf.reset_index(); final_pf = final_pf.drop(['index'], axis=1)

#logic to generate sample roster from our recommendations
while True:
    sample_pg = final_pg.sample(n=2)
    sample_pf = final_pf.sample(n=2)
    sample_sg = final_sg.sample(n=2)
    sample_sf = final_sf.sample(n=2)
    sample_roster = pd.concat([top5[['Player','SALARY','PER']],sample_pg, sample_pf, sample_sg, sample_sf], ignore_index=True)
    if (sample_roster.SALARY.sum() <= salary_cap):
        break
    else:
        continue

print("Salary cap available: "+str(avail_sal))
print()
print("PG Recommendations:")
print(final_pg)
print()
print("SG Recommendations:")
print(final_sg)
print()
print("SF Recommendations:")
print(final_sf)
print()
print("PF Recommendations:")
print(final_pf)
print()
print("Sample Roster:")
print(sample_roster)
print()
print("Sample Roster Salary: "+str(sample_roster.SALARY.sum()))


# In[29]:


#generate descriptive stats and correlation matrix for performance stats
print(stats17[cols].describe())
print(stats1316[['TS%','TRB%','AST%','TOV%','USG%','BPM','PF','PER']].describe())
plt.matshow(stats1316[['TS%','TRB%','AST%','TOV%','USG%','BPM','PF','PER']].corr())
plt.yticks(range(len(stats1316[['TS%','TRB%','AST%','TOV%','USG%','BPM','PF','PER']].columns)), stats1316[cols].columns)
plt.colorbar()
pd.plotting.scatter_matrix(stats1316[['TS%','TRB%','AST%','TOV%','USG%','BPM','PF','PER']],figsize=(10,10))
pd.plotting.scatter_matrix(stats17[cols],figsize=(10,10))
plt.show()


# In[30]:


result = pd.concat([pacers,stats17])
colist=["TS%", "PTS", "TRB", "AST", "TOV", "PF", "USG%","BPM","SALARY","G"]
new_df= result[colist]
desc_df=new_df.describe(include= "all")
desc_df[colist]=desc_df[colist].round(2)
print(desc_df.head(10))
plt.figure(figsize=(16, 6))
cust = {Tm: "teal" if Tm== "IND" else "lightblue" for Tm in result.Tm.unique()}
sal_box = sns.boxplot(x='Tm',y='SALARY', data=result, width=0.6, palette=cust)
# density plot with shade
sns.kdeplot(result['PER'],shade=True,color='lightblue',legend=False).set_title('Distribution of predicted PER')

avg_pos=result.groupby('Pos').agg({'SALARY':'mean','Tm':'count'})
avg_pos['P_Tm']=(avg_pos['Tm']/30).round(0)
avg_pos['av_sal']=(avg_pos['SALARY']/1000000).round(2)
print(avg_pos.head())
plt.show()


# In[31]:


by_pos=result.groupby('Pos').mean().reset_index()
by_pos=by_pos.sort_values(by='Pos')
sns.barplot(x='PTS',y='Pos', data=by_pos, palette='Blues')
plt.title("Season points")


# In[32]:


sns.barplot(x='TRB',y='Pos', data=by_pos, palette='Blues')
plt.title("Total Rebounds (TRB)")


# In[33]:


sns.barplot(x='AST',y='Pos', data=by_pos, palette='Blues')
plt.title("Assists (AST)")


# In[34]:


sns.barplot(x='STL',y='Pos', data=by_pos, palette='Blues')
plt.title("Steals (STL)")


# In[35]:


sns.barplot(x='TOV',y='Pos', data=by_pos, palette='Blues')
plt.title("Turnover (TOV)")


# In[36]:


sns.barplot(x='BLK',y='Pos', data=by_pos, palette='Blues')
plt.title("Blocks (BLK)")


# In[37]:


sns.barplot(x='PF',y='Pos', data=by_pos, palette='Blues')
plt.title("Personal Fouls (PF)")

