#!/usr/bin/env python
# coding: utf-8

#  
# # Capstone Project by Saravana - Airplane Data (Data_Train.xlsx)

# In[3]:


import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Airlinedata.csv")
#df.head(10)
df.shape
df.describe()

#Changing the column type from the CSV file to the appropriate data type
#Converting date of journey type to datetime format
df[['Date_of_Journey']] = df[['Date_of_Journey']].apply(pd.to_datetime,format="%d/%m/%Y") 
df["Dep_Time"] = pd.to_datetime(df["Dep_Time"],format='%H:%M')
df.info()

#Handling missing data
print("If any missing data = ",df.isnull().sum()) # Only total stops and route has one null data. We can ignore it
    


# In[168]:


#Finding Mean, Median, STD for price and duration columns

df["DurationV2"] = pd.to_timedelta(df["Duration"]) #Creating additional field in the dataframe called DurationV2 and storing the data in Timedelta format
#print(df["Duration"])

#print("Mean of airline durations grouped by Airlines",df.groupby('Airline')["DurationV2"].mean())
#print("Mean of airline Price grouped by Airlines",df.groupby('Airline')["Price"].mean())
#Mean value for Duration, Price 
mn_duration_price = {#'Airline' :df["Airline"].unique()
                     'Mean of Duration' :df.groupby('Airline')["DurationV2"].mean()
                     ,'Mean of Price':df.groupby('Airline')["Price"].mean()
                    ,'Median of Duration':df.groupby('Airline')["DurationV2"].median()
                ,'Median of Price': df.groupby('Airline')["Price"].median()
}


display(pd.DataFrame(mn_duration_price))
#############
print("Mean of all the Airline Duration & Price\n",df[["DurationV2","Price"]].mean())
print("Median of all the Airline Duration & Price\n",df[["DurationV2","Price"]].median())
price_std_dev = df['Price'].std()
duration_std_dev = df['DurationV2'].std()

print("Standard Deviation of Price:", price_std_dev)
print("Standard Deviation of Duration:", duration_std_dev)



#print("Median of airline durations grouped by Airlines",df.groupby('Airline')["DurationV2"].median())
#print("Median of airline Price grouped by Airlines",df.groupby('Airline')["Price"].median())
###############
#print(df.groupby('Airline')["DurationV2"].mean())
#print(df["Price"].mean())
#print((df["DurationV2"].median()))
#df[["DurationV2","Price"]].mean()
df.info()


# In[115]:


#Visualization of the ticket prices
ticket_prices = df["Price"]
#plt.hist(ticket_prices)
#plt.title("Distribution of ticket prices")
#plt.xlabel("Ticket Price")
#plt.ylabel("Frequency")
#plt.figure(figsize=(10, 6))
#sns.histplot(data=df,x=ticket_prices)
#plt.title("Distribution of ticket prices")
#plt.xlabel("Ticket Price")
#plt.ylabel("Frequency")
#plt.show()

plt.figure(figsize=(15, 5))

# Histogram
plt.subplot(1, 3, 1)
sns.histplot(data=df, x='Price', bins=20, kde=True)
plt.title('Price Distribution (Histogram)')

# Box plot
plt.subplot(1, 3, 2)
sns.boxplot(data=df, x='Price')
plt.title('Price Distribution (Box Plot)')

# Density plot
plt.subplot(1, 3, 3)
sns.kdeplot(data=df, x='Price', fill=True)
plt.title('Price Distribution (Density Plot)')

plt.tight_layout()
plt.show()


# In[118]:


#Finding Outliers using IQR Method
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1

# Find outliers using IQR method
outliers_iqr = df[(df['Price'] < (Q1 - 1.5 * IQR)) | (df['Price'] > (Q3 + 1.5 * IQR))]

# Find outliers using box plot visualization
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Price')
plt.title('Box Plot of Price')
plt.show()

# Display outliers found using IQR method
print("Outliers found using IQR method:")
print(outliers_iqr)


# In[131]:


#Visualization for Duration and also finding Outliers
df['DurationinHours'] = df["DurationV2"] / pd.Timedelta(hours=1)
#print(df['DurationinHours'])
# Calculate quartiles and IQR for DurationV2
Q1_duration = df['DurationinHours'].quantile(0.25)
Q3_duration = df['DurationinHours'].quantile(0.75)
IQR_duration = Q3_duration - Q1_duration

# Find outliers using IQR method for Duration in Hours
outliers_iqr_duration = df[(df['DurationinHours'] < (Q1_duration - 1.5 * IQR_duration)) | (df['DurationinHours'] > (Q3_duration + 1.5 * IQR_duration))]

# Find outliers using box plot visualization for Duration in hours
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='DurationinHours')
plt.title('Box Plot of DurationinHours')
plt.show()

# Display outliers found using IQR method for Duration in Hours
print("Outliers found using IQR method for Duration in hours:")
print(outliers_iqr_duration)

