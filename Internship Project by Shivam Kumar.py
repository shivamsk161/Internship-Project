#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')


# In[6]:


df1=pd.read_csv('Order.tsv',sep='\t')


# In[7]:


df1.head()


# In[8]:


df1.columns


# In[9]:


df2=pd.read_json('Order_breakdown.json')


# In[10]:


df2


# In[11]:


df3=pd.merge(df1,df2,on='Order ID')


# In[12]:


df3.head()


# In[13]:


df3.to_csv('df_csv.csv')


# In[14]:


df=pd.read_csv('df_csv.csv')


# In[15]:


df.head()


# In[16]:


df.drop('Unnamed: 0',axis=1,inplace=True)


# In[17]:


df.head()


# In[18]:


df.columns


# In[19]:


df.shape


# In[20]:


df.dtypes


# In[21]:


df['Order Date']=pd.to_datetime(df['Order Date'])
df['Ship Date']=pd.to_datetime(df['Ship Date'])


# In[22]:


df.dtypes


# ### Checking and handelling null values

# In[23]:


df.isnull().sum()


# ### Checking and handelling dupliactes values

# In[24]:


df.duplicated().sum()


# In[25]:


df.drop_duplicates(keep='first',inplace=True)


# In[26]:


df.duplicated().sum()


# In[27]:


df.shape


# In[28]:


df.info()


# In[29]:


df.describe()


# ### Checking and handelling outliers present in the data

# In[30]:


col_name = df.select_dtypes(include=['int','float']).columns
for i in col_name:
  mean = df[i].mean()
  med =  df[i].median()
  print(f'Mean for {i} is {mean}')
  print(f'Median for {i} is {med}')


# In[31]:


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


# In[32]:


Q1=df[col_name].quantile(0.25)
Q3=df[col_name].quantile(0.75)
IQR = Q3 - Q1
upper = Q3+(1.5*IQR)
lower = Q1-(1.5*IQR)


# In[33]:


df[col_name]=np.where(df[col_name]>upper,
                     upper,
                     np.where(df[col_name]<lower,
                     lower,
                     df[col_name]))


# In[34]:


df.describe()


# In[35]:


df.shape


# In[36]:


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

# In[37]:


df.corr()


# In[38]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),cmap='RdBu',annot=True)
plt.show()


# In[39]:


def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if(corr_matrix.iloc[i,j])>threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return list(col_corr)        


# In[40]:


corr_features=correlation(df,0.7)
len(set(corr_features))


# In[41]:


corr_features


# In[42]:


df.drop(columns=corr_features,axis=1,inplace=True)


# In[43]:


df.shape


# In[44]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),cmap='RdBu',annot=True)
plt.show()


# In[45]:


df.drop('Order ID',axis=1,inplace=True)


# In[46]:


df.head()


# ### EDA

# In[47]:


plt.figure(figsize=(5, 5))
sns.barplot(x=df['Category'],y=df['Sales'],ci=None)
plt.xticks(rotation='vertical')
plt.show()


# #### Conclusion:'Technology' Category has more number of sales compared to other Categories available in the data.

# In[48]:


sns.countplot(df['Ship Mode'])


# #### Conclusion:Mostly 'Economy' class ship mode is prefered from rest of the other available ship mode.

# In[49]:


sales_by_subcategory=df.groupby('Sub-Category')['Sales'].sum()


# In[53]:


sales_by_subcategory.plot(kind='bar')
plt.title('Sales by Sub-Category')
plt.xlabel('Sub-Category')
plt.ylabel('Sales')
plt.show()


# #### Conclusion:Sales in 'Storage' in sub-category is maximum.

# In[54]:


sales_by_country=df.groupby('Country')['Sales'].sum()


# In[56]:


sales_by_country.plot(kind='bar')
plt.title('Sales by Country')
plt.xlabel('Country')
plt.ylabel('Sales')
plt.show()


# ### Conclusion:Most sales were made by 'France' .

# In[57]:


sales_by_region=df.groupby('Region')['Sales'].sum()


# In[63]:


sales_by_region.plot(kind='pie',autopct='%.2f%%')
plt.title('Sales by Region')
plt.show()


# #### Conclusion:'Central' region has done the maximum number of sales.

# In[148]:


sns.countplot(df['Category'])


# #### Conclusion:Mostly 'Office Supplies' have been ordered in majority from all other categories available.

# In[149]:


sns.countplot(df['Region'])


# #### Conclusion:Most of the orders were from 'Central Region'.

# In[150]:


plt.figure(figsize=(8,4))
sns.countplot(df['Sub-Category'])
plt.xticks(rotation=90)
plt.show()


# #### Conclusion:Mostly 'Art' items were ordered from the Sub-Category.

# In[151]:


plt.figure(figsize=(8,4))
sns.countplot(df['Country'])
plt.xticks(rotation=90)
plt.show()


# #### Conclusion:Majority of the orders were from France and Germany.

# In[152]:


sns.barplot(df['Category'],df['Profit'],ci=None)


# #### Conlusion:Mostly 'Technology' sector  from Category has gathered more Profit from rest of the other Categories available. 

# In[153]:


sns.distplot(df['Sales'])


# In[154]:


sns.displot(df['Profit'])


# ### Selecting dependent and independent features

# In[155]:


x=df.drop(['Sales'],axis=1)
y=df['Sales']


# In[156]:


print(x.shape)
print(y.shape)
print(type(x))
print(type(y))


# In[157]:


x.head()


# In[158]:


y.head()


# In[159]:


from sklearn.model_selection import train_test_split


# In[160]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)
print(type(x_train),type(x_test))
print(type(y_train),type(y_test))


# In[161]:


x_train.head()


# In[162]:


y_train.head()


# ### Scalling Data

# #### Scalling categorical data of x_train

# In[163]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[164]:


x_train_cat =x_train.select_dtypes(include=['object'])
x_train_cat.head()


# In[165]:


x_train_cat.columns


# In[166]:


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


# In[167]:


x_train_cat


# ## Numerical Data of x_train

# In[168]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[169]:


x_train_num = x_train.select_dtypes(include=['int64', 'float64'])
x_train_num.head()


# In[170]:


x_train_num_ss=pd.DataFrame(sc.fit_transform(x_train_num),columns=x_train_num.columns, index=x_train_num.index)
x_train_num_ss.head()


# ## Concatinating Cat and Num dataframes

# In[171]:


x_train_rescaled =pd.concat([x_train_cat,x_train_num_ss], axis=1)
x_train_rescaled.head()


# # Scalling x_test data

# ### Scalling Categorical Data of x_test

# In[172]:


x_test_cat=x_test.select_dtypes(include=['object'])
x_test_cat.head()


# In[173]:


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


# In[174]:


x_test_cat


# ## Numerical Data of x_test

# In[175]:


x_test_num=x_test.select_dtypes(include=['int64', 'float64'])
x_test_num.head()


# In[176]:


x_test_num_ss=pd.DataFrame(sc.fit_transform(x_test_num),columns=x_test_num.columns,index=x_test_num.index)
x_test_num_ss


# ## Concatinating Cat and Num dataframes

# In[177]:


x_test_rescaled =pd.concat([x_test_cat,x_test_num_ss], axis=1)
x_test_rescaled.head()


# ### Creating functions to evaluate confusion matrix,classification report,accuracy score

# In[178]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[179]:


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

# In[180]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso


# In[181]:


lr=LinearRegression()


# In[182]:


lr.fit(x_train_rescaled,y_train)


# In[183]:


ypred_lr=lr.predict(x_test_rescaled)


# In[184]:


reg_eval_metrics(y_test,ypred_lr)


# In[185]:


from sklearn.neighbors import KNeighborsRegressor


# In[186]:


kn=KNeighborsRegressor()


# In[187]:


kn.fit(x_train_rescaled,y_train)


# In[188]:


ypred_kn=kn.predict(x_test_rescaled)


# In[189]:


reg_eval_metrics(y_test,ypred_kn)


# In[190]:


ridge=Ridge()


# In[191]:


ridge.fit(x_train_rescaled,y_train)


# In[192]:


ypred_ridge=ridge.predict(x_test_rescaled)


# In[193]:


reg_eval_metrics(y_test,ypred_ridge)


# In[194]:


from sklearn.ensemble import GradientBoostingRegressor


# In[195]:


gb= GradientBoostingRegressor()


# In[196]:


gb.fit(x_train_rescaled,y_train)


# In[197]:


ypred_gb=gb.predict(x_test_rescaled)


# In[198]:


reg_eval_metrics(y_test,ypred_gb)


# In[199]:


print('Training Score',gb.score(x_train_rescaled,y_train))
print('Testing Score',gb.score(x_test_rescaled,y_test))


# In[200]:


from sklearn.tree import DecisionTreeRegressor


# In[201]:


dt=DecisionTreeRegressor(criterion='squared_error',splitter='best',max_depth=None,min_samples_split=10)


# In[202]:


dt.fit(x_train_rescaled,y_train)


# In[203]:


ypred_dt=dt.predict(x_test_rescaled)


# In[204]:


reg_eval_metrics(y_test,ypred_dt)


# In[205]:


print('Training Score',dt.score(x_train_rescaled,y_train))
print('Testing Score',dt.score(x_test_rescaled,y_test))


# In[206]:


from sklearn.ensemble import RandomForestRegressor


# In[207]:


rf=RandomForestRegressor(n_estimators=100,min_samples_split=2)


# In[208]:


rf.fit(x_train_rescaled,y_train)


# In[209]:


ypred_rf=rf.predict(x_test_rescaled)


# In[210]:


reg_eval_metrics(y_test,ypred_rf)


# In[211]:


print('Training Score',rf.score(x_train_rescaled,y_train))
print('Testing Score',rf.score(x_test_rescaled,y_test))


# ### Inference

# #### Based on accuracy,Gradient Boosting Regressor performs better compared to other models getting an Training score of 81%.

# ### Saving the Gradient Boosting Regressor model

# ### Dumping the model

# In[99]:


from pickle import dump


# In[100]:


dump(lb,open('label_encoder.pkl','wb'))
dump(sc,open('standard_scaler.pkl','wb'))
dump(gb,open('gb.pkl','wb'))


# ### Loading the model

# In[101]:


from pickle import load


# In[102]:


lb=load(open('label_encoder.pkl','rb'))
sc=load(open('standard_scaler.pkl','rb'))
gb=load(open('gb.pkl','rb'))


# In[103]:


test_acc=gb.score(x_test_rescaled,y_test)


# In[104]:


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

