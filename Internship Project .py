#!/usr/bin/env python
# coding: utf-8

# In[205]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')


# In[206]:


df1=pd.read_csv('Order.tsv',sep='\t')


# In[207]:


df1


# In[208]:


df2=pd.read_json('Order_breakdown.json')


# In[209]:


df2.head()


# In[210]:


df3=pd.merge(df1,df2,on='Order ID')


# In[211]:


df3.head()


# In[212]:


df3.to_csv('df_csv.csv')


# In[213]:


df=pd.read_csv('df_csv.csv')


# In[214]:


df.head()


# In[215]:


df.drop('Unnamed: 0',axis=1,inplace=True)


# In[216]:


df.head()


# In[217]:


df.columns


# In[218]:


df.shape


# In[219]:


df.dtypes


# In[220]:


df['Order Date']=pd.to_datetime(df['Order Date'])
df['Ship Date']=pd.to_datetime(df['Ship Date'])


# In[221]:


df.dtypes


# ### Checking and handelling null values

# In[222]:


df.isnull().sum()


# ### Checking and handelling dupliactes values

# In[223]:


df.duplicated().sum()


# In[224]:


df.drop_duplicates(keep='first',inplace=True)


# In[225]:


df.duplicated().sum()


# In[226]:


df.shape


# In[227]:


df.info()


# In[228]:


df.describe()


# ### Checking and handelling outliers present in the data

# In[229]:


col_name = df.select_dtypes(include=['int','float']).columns
for i in col_name:
  mean = df[i].mean()
  med =  df[i].median()
  print(f'Mean for {i} is {mean}')
  print(f'Median for {i} is {med}')


# In[230]:


fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(15,10))
ax1.boxplot(df['Discount'])
ax1.set_title('Boxplot for Discount')
ax2.boxplot(df['Actual Discount'])
ax2.set_title('Boxplot for Actual Discount')
ax3.boxplot(df['Sales'])
ax3.set_title('Boxplot for Sales')
ax4.boxplot(df['Profit'])
ax4.set_title('Boxplot for Profit')
plt.show()


# In[231]:


Q1=df[col_name].quantile(0.25)
Q3=df[col_name].quantile(0.75)
IQR = Q3 - Q1
upper = Q3+(1.5*IQR)
lower = Q1-(1.5*IQR)


# In[232]:


df[col_name]=np.where(df[col_name]>upper,
                     upper,
                     np.where(df[col_name]<lower,
                     lower,
                     df[col_name]))


# In[233]:


df.describe()


# In[234]:


df.shape


# In[235]:


fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(15,10))
ax1.boxplot(df['Discount'])
ax1.set_title('Boxplot for Discount')
ax2.boxplot(df['Actual Discount'])
ax2.set_title('Boxplot for Actual Discount')
ax3.boxplot(df['Sales'])
ax3.set_title('Boxplot for Sales')
ax4.boxplot(df['Profit'])
ax4.set_title('Boxplot for Profit')
plt.show()


# ### Checking and handelling correlated values

# In[236]:


df.corr()


# In[237]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),cmap='RdBu',annot=True)
plt.show()


# In[238]:


def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if(corr_matrix.iloc[i,j])>threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return list(col_corr)        


# In[239]:


corr_features=correlation(df,0.7)
len(set(corr_features))


# In[240]:


corr_features


# In[241]:


df.drop(columns=corr_features,axis=1,inplace=True)


# In[242]:


df.shape


# In[243]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),cmap='RdBu',annot=True)
plt.show()


# In[244]:


df.drop('Order ID',axis=1,inplace=True)


# In[245]:


df.head()


# ### Selecting dependent and independent features

# In[246]:


x=df.drop(['Sales'],axis=1)
y=df['Sales']


# In[247]:


print(x.shape)
print(y.shape)
print(type(x))
print(type(y))


# In[248]:


x.head()


# In[249]:


y.head()


# In[250]:


from sklearn.model_selection import train_test_split


# In[251]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)
print(type(x_train),type(x_test))
print(type(y_train),type(y_test))


# In[252]:


x_train.head()


# In[253]:


y_train.head()


# ### Scalling Data

# #### Scalling categorical data of x_train

# In[254]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[255]:


x_train_cat =x_train.select_dtypes(include=['object'])
x_train_cat.head()


# In[256]:


x_train_cat.columns


# In[257]:


x_train_cat['Customer Name']=lb.fit_transform(x_train_cat['Customer Name'])
x_train_cat['City']=lb.fit_transform(x_train_cat['City'])
x_train_cat['Country']=lb.fit_transform(x_train_cat['Country'])
x_train_cat['Region']=lb.fit_transform(x_train_cat['Region'])
x_train_cat['Segment']=lb.fit_transform(x_train_cat['Segment'])
x_train_cat['Ship Mode']=lb.fit_transform(x_train_cat['Ship Mode'])
x_train_cat['State']=lb.fit_transform(x_train_cat['State'])
x_train_cat['Product Name']=lb.fit_transform(x_train_cat['Product Name'])
x_train_cat['Category']=lb.fit_transform(x_train_cat['Category'])
x_train_cat['Sub-Category']=lb.fit_transform(x_train_cat['Sub-Category'])


# In[258]:


x_train_cat


# ## Numerical Data of x_train

# In[259]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[260]:


x_train_num = x_train.select_dtypes(include=['int64', 'float64'])
x_train_num.head()


# In[261]:


x_train_num_ss=pd.DataFrame(sc.fit_transform(x_train_num),columns=x_train_num.columns, index=x_train_num.index)
x_train_num_ss.head()


# ## Concatinating Cat and Num dataframes

# In[262]:


x_train_rescaled =pd.concat([x_train_cat,x_train_num_ss], axis=1)
x_train_rescaled.head()


# # Scalling x_test data

# ### Scalling Categorical Data of x_test

# In[263]:


x_test_cat=x_test.select_dtypes(include=['object'])
x_test_cat.head()


# In[264]:


x_test_cat['Customer Name']=lb.fit_transform(x_test_cat['Customer Name'])
x_test_cat['City']=lb.fit_transform(x_test_cat['City'])
x_test_cat['Country']=lb.fit_transform(x_test_cat['Country'])
x_test_cat['Region']=lb.fit_transform(x_test_cat['Region'])
x_test_cat['Segment']=lb.fit_transform(x_test_cat['Segment'])
x_test_cat['Ship Mode']=lb.fit_transform(x_test_cat['Ship Mode'])
x_test_cat['State']=lb.fit_transform(x_test_cat['State'])
x_test_cat['Product Name']=lb.fit_transform(x_test_cat['Product Name'])
x_test_cat['Category']=lb.fit_transform(x_test_cat['Category'])
x_test_cat['Sub-Category']=lb.fit_transform(x_test_cat['Sub-Category'])


# In[265]:


x_test_cat


# ## Numerical Data of x_test

# In[266]:


x_test_num=x_test.select_dtypes(include=['int64', 'float64'])
x_test_num.head()


# In[267]:


x_test_num_ss=pd.DataFrame(sc.fit_transform(x_test_num),columns=x_test_num.columns,index=x_test_num.index)
x_test_num_ss


# ## Concatinating Cat and Num dataframes

# In[268]:


x_test_rescaled =pd.concat([x_test_cat,x_test_num_ss], axis=1)
x_test_rescaled.head()


# ### Creating functions to evaluate confusion matrix,classification report,accuracy score

# In[269]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[270]:


def reg_eval_metrics(ytest, ypred): 
    mae = mean_absolute_error(ytest, ypred)
    mse = mean_squared_error(ytest, ypred)
    rmse = np.sqrt(mse)
    r2 = r2_score(ytest, ypred)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

def train_test_scr(model):
    print('Training Score',model.score(x_train_rescaled,y_train_rescaled))  # R2 score for Training data
    print('Testing Score',model.score(x_test_rescaled,y_test_rescaled))     # R2 score for test data


# ## Model Training

# In[271]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso


# In[272]:


lr=LinearRegression()


# In[273]:


lr.fit(x_train_rescaled,y_train)


# In[274]:


ypred_lr=lr.predict(x_test_rescaled)


# In[275]:


reg_eval_metrics(y_test,ypred_lr)


# In[276]:


from sklearn.neighbors import KNeighborsRegressor


# In[277]:


kn=KNeighborsRegressor()


# In[278]:


kn.fit(x_train_rescaled,y_train)


# In[279]:


ypred_kn=kn.predict(x_test_rescaled)


# In[280]:


reg_eval_metrics(y_test,ypred_kn)


# In[281]:


ridge=Ridge()


# In[282]:


ridge.fit(x_train_rescaled,y_train)


# In[283]:


ypred_ridge=ridge.predict(x_test_rescaled)


# In[284]:


reg_eval_metrics(y_test,ypred_ridge)


# In[285]:


from sklearn.ensemble import GradientBoostingRegressor


# In[286]:


gb= GradientBoostingRegressor()


# In[287]:


gb.fit(x_train_rescaled,y_train)


# In[288]:


ypred_gb=gb.predict(x_test_rescaled)


# In[289]:


reg_eval_metrics(y_test,ypred_gb)


# In[290]:


print('Training Score',gb.score(x_train_rescaled,y_train))
print('Testing Score',gb.score(x_test_rescaled,y_test))


# In[291]:


from sklearn.tree import DecisionTreeRegressor


# In[292]:


dt=DecisionTreeRegressor(criterion='squared_error',splitter='best',max_depth=None,min_samples_split=10)


# In[293]:


dt.fit(x_train_rescaled,y_train)


# In[294]:


ypred_dt=dt.predict(x_test_rescaled)


# In[295]:


reg_eval_metrics(y_test,ypred_dt)


# In[296]:


print('Training Score',dt.score(x_train_rescaled,y_train))
print('Testing Score',dt.score(x_test_rescaled,y_test))


# In[297]:


from sklearn.ensemble import RandomForestRegressor


# In[298]:


rf=RandomForestRegressor(n_estimators=100,min_samples_split=2)


# In[299]:


rf.fit(x_train_rescaled,y_train)


# In[300]:


ypred_rf=rf.predict(x_test_rescaled)


# In[301]:


reg_eval_metrics(y_test,ypred_rf)


# In[302]:


print('Training Score',rf.score(x_train_rescaled,y_train))
print('Testing Score',rf.score(x_test_rescaled,y_test))


# ### Inference

# #### Based on accuracy,Gradient Boosting Regressor performs better compared to other models getting an Training score of 81%.

# ### Saving the Gradient Boosting Regressor model

# ### Dumping the model

# In[303]:


from pickle import dump


# In[304]:


dump(lb,open('label_encoder.pkl','wb'))
dump(sc,open('standard_scaler.pkl','wb'))
dump(gb,open('gb.pkl','wb'))


# ### Loading the model

# In[305]:


from pickle import load


# In[306]:


lb=load(open('label_encoder.pkl','rb'))
sc=load(open('standard_scaler.pkl','rb'))
gb=load(open('gb.pkl','rb'))


# In[307]:


test_acc=gb.score(x_test_rescaled,y_test)


# In[308]:


test_acc


# In[309]:


df.head()


# In[310]:


x_train_rescaled.head()


# In[312]:


name = input("enter customer name : ")
city = input("enter city : ")
country= input("enter country : ")
region = input("enter region : ")
segment = input("enter segment : ")
ship=input("enter ship mode : ")
state=input("enter state : ")
prod=input("enter product name : ")
cat=input("enter category : ")
sub_cat=input("enter sub-category : ")
day_to_ship=float(input("enter the days to ship : "))
discount=float(input("enter the discount : "))
profit=float(input("enter the profit : "))
quantity=float(input("enter the quantity : "))


# In[ ]:


Ruby Patel Stockholm Sweden North Home Office Economy Plus Stockholm Enermax Note Cards, Premium Office Supplies Paper 4.0 0.25-26.0 


# In[313]:


x_train_rescaled.columns


# In[314]:


query_cat=pd.DataFrame({'Customer Name':[name], 'City':[city],'Country':[country],'Region':[region],'Segment':[segment],'Ship Mode':[ship],'State':[state],'Product Name':[prod],'Category':[cat],'Sub-Category':[sub_cat]})
query_num=pd.DataFrame({'Days to Ship':[day_to_ship], 'Discount':[discount],'Profit':[profit],'Quantity':[quantity]})


# In[315]:


query_cat


# In[316]:


query_num


# In[317]:


query_cat['Customer Name']=lb.fit_transform(query_cat['Customer Name'])
query_cat['City']=lb.fit_transform(query_cat['City'])
query_cat['Country']=lb.fit_transform(query_cat['Country'])
query_cat['Region']=lb.fit_transform(query_cat['Region'])
query_cat['Segment']=lb.fit_transform(query_cat['Segment'])
query_cat['Ship Mode']=lb.fit_transform(query_cat['Ship Mode'])
query_cat['State']=lb.fit_transform(query_cat['State'])
query_cat['Product Name']=lb.fit_transform(query_cat['Product Name'])
query_cat['Category']=lb.fit_transform(query_cat['Category'])
query_cat['Sub-Category']=lb.fit_transform(query_cat['Sub-Category'])


# In[318]:


query_cat


# In[319]:


query_num=pd.DataFrame(sc.fit_transform(query_num),columns=query_num.columns, index=query_num.index)
query_num.head()


# In[320]:


query=pd.concat([pd.DataFrame(query_cat),pd.DataFrame(query_num)],axis=1)


# In[321]:


query


# In[322]:


Sales=gb.predict(query)


# In[325]:


print(f"Sales is {round(Sales[0],0)}")


# In[326]:


df4=df.copy()


# In[328]:


df4.to_csv('Sales_df.csv')


# ### Getting samples from dataset

# In[329]:


sample=df.sample(20)


# In[330]:


sample


# In[331]:


sample.to_csv('sample.csv')


# In[332]:


sample.head()


# In[333]:


a=sample.drop(['Sales'],axis=1)
b=sample['Sales']


# In[334]:


a.head()


# In[335]:


b.head()


# In[336]:


a.shape


# In[337]:


b.shape


# ### Scalling

# In[338]:


a_cat =a.select_dtypes(include=['object'])
a_cat.head()


# In[339]:


a_cat['Customer Name']=lb.fit_transform(a_cat['Customer Name'])
a_cat['City']=lb.fit_transform(a_cat['City'])
a_cat['Country']=lb.fit_transform(a_cat['Country'])
a_cat['Region']=lb.fit_transform(a_cat['Region'])
a_cat['Segment']=lb.fit_transform(a_cat['Segment'])
a_cat['Ship Mode']=lb.fit_transform(a_cat['Ship Mode'])
a_cat['State']=lb.fit_transform(a_cat['State'])
a_cat['Product Name']=lb.fit_transform(a_cat['Product Name'])
a_cat['Category']=lb.fit_transform(a_cat['Category'])
a_cat['Sub-Category']=lb.fit_transform(a_cat['Sub-Category'])


# In[340]:


a_cat


# In[341]:


a_num = a.select_dtypes(include=['int64', 'float64'])
a_num.head()


# In[342]:


a_num_ss=pd.DataFrame(sc.fit_transform(a_num),columns=a_num.columns, index=a_num.index)
a_num_ss.head()


# In[343]:


a_rescaled =pd.concat([a_cat,a_num_ss], axis=1)
a_rescaled.head()


# In[344]:


a_rescaled.columns


# In[345]:


x_train_rescaled.columns


# In[346]:


pred=gb.predict(a_rescaled)


# In[347]:


pred


# In[348]:


model={'Actual value':b,
       'Predicted value':pred}


# In[349]:


pred=pd.DataFrame(model)


# In[350]:


pred


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




