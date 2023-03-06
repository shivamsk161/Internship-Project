import streamlit as st
import pandas as pd
from pickle import load

lb=load(open('label_encoder.pkl','rb'))
sc=load(open('standard_scaler.pkl','rb'))
gb=load(open('gb.pkl','rb'))

st.title('Sales Prediction')

df=pd.read_csv('Sales_df.csv')

with st.form('my_form'):
  Name=st.selectbox(label='Customer Name',options=df['Customer Name'].unique())
  City=st.selectbox(label='City',options=df['City'].unique())
  Country=st.selectbox(label='Country',options=df['Country'].unique())
  Region=st.selectbox(label='Region',options=df['Region'].unique())
  Segment=st.selectbox(label='Segment',options=df['Segment'].unique())
  Ship_Mode=st.selectbox(label='Ship_Mode',options=df['Ship Mode'].unique())
  State=st.selectbox(label='State',options=df['State'].unique())
  Product_name=st.selectbox(label='Product_name',options=df['Product Name'].unique())
  Category=st.selectbox(label='Category',options=df['Category'].unique())
  Sub_Category=st.selectbox(label='Sub_Category',options=df['Sub-Category'].unique())
  Days_to_Ship=st.select_slider(label='Days_to_Ship',options=df['Days to Ship'].sort_values())
  Discount=st.select_slider(label='Discount',options=df['Discount'].sort_values())
  Profit=st.select_slider(label='Profit',options=df['Profit'].sort_values())
  Quantity=st.select_slider(label='Quantity',options=df['Quantity'].sort_values())
  
  btn = st.form_submit_button(label='Predict')
  
  if btn:
    if Name and City and Country and Region and Segment and Ship_Mode and State and Product_name and Category and Sub_Category and Days_to_Ship and Discount and Profit and Quantity:
      query_cat=pd.DataFrame({'Customer Name':[name], 'City':[city],'Country':[country],'Region':[region],'Segment':[segment],'Ship Mode':[ship],'State':[state],'Product Name':[prod],'Category':[cat],'Sub-Category':[sub_cat]})
      query_num=pd.DataFrame({'Days to Ship':[day_to_ship], 'Discount':[discount],'Profit':[profit],'Quantity':[quantity]})
      
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
      
      query_num=pd.DataFrame(sc.fit_transform(query_num),columns=query_num.columns, index=query_num.index)
      
      query=pd.concat([pd.DataFrame(query_cat),pd.DataFrame(query_num)],axis=1)
      Sales=gb.predict(query)       
      st.success(f"Sales is {round(Sales[0],0)}")
      
      else:
      st.error('Category not found')
