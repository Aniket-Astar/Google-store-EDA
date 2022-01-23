#!/usr/bin/env python
# coding: utf-8

# #  Importing the required packages

# In[80]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #  Reading Data

# In[81]:


googlestore_df1 = pd.read_csv("googleplaystore.csv")
googlestore_df2 = pd.read_csv("googleplaystore_user_reviews.csv")


# In[82]:


googlestore_df1.head()


# In[83]:


googlestore_df2.head()


# In[84]:


rows =googlestore_df1.shape[0]
column = googlestore_df1.shape[1]


# In[85]:


print('There are {} Rows and {} colums in the dataset'.format(rows, column))


# In[86]:


googlestore_df1.info()


# # Data Processing
# #  Any Null value?    

# In[87]:


googlestore_df1.isnull().sum()


# #  Removing Duplicates Entries

# In[88]:


googlestore_df1.shape


# # Data Types

# In[89]:


googlestore_df1.dtypes #Displaying Data types


# In[90]:


googlestore_df1.Category.value_counts()


# In[91]:


googlestore_df1.columns


# #  Rows of the column Rating having NaN values
# 

# In[92]:


googlestore_df1[googlestore_df1.Rating.isnull()]


# # Rows of the column Type having NULL values

# In[93]:


googlestore_df1[googlestore_df1.Type.isnull()]


# In[94]:


# Since there is only one value missing in the Type column
# So, Lets fill the missing value, After cross-checking in the playstore the missing value is found to be Free, So Now we can fill the missing value with Free


# # Fill missing rating with mean

# In[95]:


googlestore_df1['Rating'] = googlestore_df1['Rating'].fillna(round(googlestore_df1['Rating'].mean(),1))
googlestore_df1.dropna(inplace = True)


# In[96]:


googlestore_df1.isnull().sum()


# # Rows column content Rating having NULL Values

# In[97]:


googlestore_df1[googlestore_df1['Content Rating'].isnull()]


# In[98]:


googlestore_df1['Reviews'] = googlestore_df1['Reviews'].astype('int')


# In[99]:


googlestore_df1['Installs'].unique()


# # 

# In[100]:


googlestore_df1['Installs'] = googlestore_df1['Installs'].apply(lambda x: x.replace(',','') if ',' in x else x)
googlestore_df1['Installs'] = googlestore_df1['Installs'].apply(lambda x: x.replace('+','') if '+' in x else x)
googlestore_df1['Installs'] = googlestore_df1['Installs'].astype(float)
googlestore_df1.head()                                                           
       


# # Coverting size into same units

# In[101]:


googlestore_df1['Size'].unique()


# In[102]:


def convert_size_in_kb(x):
    if x == 'Varies with device':
        return np.NaN
    else:
        return x.replace('[A-za-z]','',regex = True)


# In[103]:


googlestore_df1 = googlestore_df1.replace({'Size': 'Varies with device'}, np.NaN)
googlestore_df1.head()


# # Changing the feature, Price

# In[104]:


newPrice = []

for row in googlestore_df1.Price:
    if row!= "0":
        newrow = float(row[1:])
    else:
        newrow = 0 
        
    newPrice.append(newrow)
        
googlestore_df1.Price = newPrice

googlestore_df1.Price.head()
    


# # Changing the feature, Android Ver

# In[105]:


newVer = []

for row in googlestore_df1['Android Ver']:
    try:
        newrow = float(row[:2])
    except:
        newrow = 0  # When the value is - Varies with device
    
    newVer.append(newrow)
    
googlestore_df1['Android Ver'] =  newVer

googlestore_df1['Android Ver'].value_counts()


# In[106]:


googlestore_df1.Category.value_counts()


# In[ ]:





# # Data viz & Analysis

# In[107]:


googlestore_df1.head(4)


# In[108]:


googlestore_df1.Category.value_counts().plot(kind='barh',figsize= (12,8))


# Insight: Maximum number of apps belong to the family and game Category.

# # Rating

# 

# In[109]:


googlestore_df1.Rating.describe()


# Distribution plot of 'Rating'

# In[110]:


sns.distplot(googlestore_df1.Rating)


# 

# # Type

# In[111]:


googlestore_df1.Type.value_counts()


# In[112]:


fig = plt.figure(figsize =(10,8))
plt.pie(googlestore_df1.Type.value_counts(), labels = ['Free','Paid'], autopct = '%1.1f%%')


# Insight: 93% of the Apps are free in the play store

# In[113]:


plt.hist(googlestore_df1.Type, color = 'green', align = 'left')
plt.title('Type')
plt.ylabel('Numbers of Apps')
plt.show()


# In[114]:


googlestore_df1['Content Rating'].value_counts()


# In[115]:


x2 = googlestore_df1['Content Rating'].value_counts().index
y2 = googlestore_df1['Content Rating'].value_counts()
x2sis = []
y2sis = []
for i in range(len(x2)):
    x2sis.append(x2[i])
    y2sis.append(y2[i])


# In[116]:


plt.figure(figsize=(12,10))
plt.bar(x2sis,y2sis,width=0.8,color=['#15244C','#FFFF48','#292734','#EF2920','#CD202D','#ECC5F2'], alpha=0.8);
plt.title('Content Rating',size = 20);
plt.ylabel('Apps(Count)');
plt.xlabel('Content Rating');


# In[118]:


fig = plt.figure(figsize =(15,8))
plt.pie(googlestore_df1['Content Rating'].value_counts(), labels = ['Everyone','Teen','Mature 17+','Everyone 10+','Adults only 18+','Unrated'], autopct = '%1.1f%%')


# # MAx price app name

# In[119]:


googlestore_df1[googlestore_df1.Price == googlestore_df1.Price.max()]


# In[120]:


googlestore_df1['Android Ver'].value_counts()


# In[121]:


sns.countplot(googlestore_df1['Android Ver'])


# # Highest Rated Category

# In[122]:


category_group = googlestore_df1.groupby('Category')
categories = googlestore_df1.Category.unique()


# In[123]:


rating = pd.DataFrame(round(category_group['Rating'].mean(),2).sort_values(ascending = False))


# In[124]:


fig = plt.figure(figsize = (12,6))
plt.bar(rating.index, rating['Rating'], color = 'Orange')
plt.xlabel('Category')
plt.xticks(rotation = 90)
plt.ylabel('Avg. Rating')
plt.yticks(range(0,6,1))
plt.title('Category vs Rating')
plt.show()
fig.savefig('Category vs Rating')


# In[125]:


plt.figure(figsize=(15,9))
plt.xlabel("Rating")
plt.ylabel("Frquency")
graph = sns.kdeplot(googlestore_df1.Rating, color ='blue' ,shade = True)
plt.title('Distrubution of Rating',size = 20)


# # Category with Highest number of Total Installs

# In[126]:


total_installs = pd.DataFrame(category_group['Installs'].sum().sort_values()/1000000)


# In[127]:


fig = plt.figure(figsize = (12,12))
plt.barh(total_installs.index, total_installs['Installs'], color = 'pink')
plt.ylabel('Category')
plt.xlabel('Total Installs in Millions')
plt.title('Category vs Total Installs in Millions')
plt.show()
fig.savefig('Category vs Total Installs in Millions')
                 


# # # Top 10 installed apps in game category

# In[133]:


def findtop10incategory(str):
    str = str.upper()
    top10 = googlestore_df1[googlestore_df1['Category'] == str]
    top10apps = top10.sort_values(by = 'Installs', ascending = False).head(10)
    
    #Top_Apps_in_art_and_design
    plt.figure(figsize =(15,8))
    plt.title('Top 10 installed Apps', size = 10);
    graph = sns.barplot(x = top10apps.App, y = top10apps.Installs)
    graph.set_xticklabels(graph.get_xticklabels(),rotation = 45, horizontalalignment = 'right');
    
    
    


# In[134]:


findtop10incategory('Sports')


# # Free vs Paid Category
# 

# In[135]:


type_grp = googlestore_df1.groupby('Type')


# In[136]:


free_grp = type_grp.get_group('Free').groupby('Category')
free_grp = round((free_grp.size()/category_group.size())*100,1)


# In[137]:


paid_grp = type_grp.get_group('Paid').groupby('Category')
paid_grp = round((paid_grp.size()/category_group.size())*100,1)


# In[138]:


fig = plt.figure(figsize = (12,14))
plt.barh(free_grp.index, free_grp, color = 'DarkGreen')
plt.barh(free_grp.index, paid_grp, left =free_grp, color = 'y')
plt.xlabel('percentage %')
plt.xticks(range(0,110,10))
plt.ylabel('Category')
plt.title('% Free vs Paid by Category')
plt.show()
fig.savefig('% Free vs Paid by Category')


# # Analysis by: Aniket Belokar Jain 

# In[ ]:





# In[ ]:




