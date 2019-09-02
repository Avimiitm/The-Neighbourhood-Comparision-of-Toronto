#!/usr/bin/env python
# coding: utf-8

# ## Segmenting and Clustering Neighbourhood of Toronto City
# 

# ### Introduction
# In this project will be using foursqure API to explore neighbourhood of Toronto city . We will use explore function of the API to get most common venues categories in each neighbourhood. Then by finding the features of the neighbourhood we will group the neighbourhood into cluster by using K-Means cluster algorithm. And finally we will use the Folium library to visualize the neighborhoods in Toronto City and their emerging clusters. We will use wikipedia table as data for this project.
# 
# ### Table of Contents
# ###### Explore Data Set
# ###### Explore Neighbourhood in Toronto City
# ###### Analyze each Neighbourhood
# ###### Clustering neighbourhood
# ###### Examine Clusters
# ###### Getting data from wikipedia_page

# In[4]:


import numpy as np

import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json 

get_ipython().system('conda install -c conda-forge geopy --yes')
from geopy.geocoders import Nominatim

import requests 
from pandas.io.json import json_normalize 

import matplotlib.cm as cm
import matplotlib.colors as colors

from sklearn.cluster import KMeans

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium 

from bs4 import BeautifulSoup

print('Libraries imported.')


# In[5]:


sourse = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text
#sourse


# In[6]:


html_data  = BeautifulSoup(sourse,"html.parser")


# In[7]:


table = html_data.find('table',class_ ="wikitable sortable")
#table


# #### Creating a List to store table data

# In[8]:


table_list =[]
for td in table.find_all('td'):
        table_list.append(td.text)
print(len(table_list))  


# #### Now stroing this table_list data into a dictionary

# In[9]:


i=0
k=0
d ={}
tdd = table_list
while(i<len(tdd)):
    Postcode = tdd[i]
    Borough = tdd[i+1]
    Neighbourhood = tdd[i+2]
    #print(Postcode,Borough,Neighbourhood)
    d[k] = [Postcode,Borough,Neighbourhood]
    i = i+3
    k +=1


# ##### Transform this dictionary into dataframe and clean the data

# In[10]:


df = pd.DataFrame(d)
df1 = df.T
# renaming columns
df1.rename(columns ={0:'PostCode',1:'Borough',2:'Neighbourhood'},inplace =True)
df1=df1[df1.Borough !='Not assigned']
df2 = df1.reset_index(drop =True)
#Removing the '\n' and ASCII symbols
df3 = df2['Neighbourhood'].values
for i in range(0,len(df3)):
    df3[i]=df3[i].replace('\n','')
df2['Neighbourhood'] = df3    
df_table = df2.groupby(['PostCode','Borough'])['Neighbourhood'].apply(','.join).reset_index()
df_table.head()


# #### Now we will use 'Geospatial_Coordinate.csv' file to get the Latitude & Longitude of the Neighbors and merge it with df_table data

# In[12]:


df_ll = pd.read_csv('Geospatial_Coordinates.csv')
df_table.rename(columns ={'PostCode':'Postal Code'},inplace =True)
df_f = pd.merge(df_table,df_ll,on ='Postal Code')
df_f.head()


# #### Use Geolocator to get latitude and longitude of Toronto

# In[13]:


adress = 'Toronto, Ontario'
geol = Nominatim() 
loc = geol.geocode(adress)
lat =loc.latitude
lng = loc.longitude
print('The geographical corinate of Toronto city are {},{}'.format(lat,lng))


# #### Let's visualize the neighborhoods of Toronto city

# In[14]:


map_toronto = folium.Map(location =[lat,lng],zoom_start = 10)

for lat1,lng1,bor,nei in zip(df_f['Latitude'],df_f['Longitude'],df_f['Borough'],df_f['Neighbourhood']):
    label = '{},{}'.format(nei,bor)
    label = folium.Popup(label,parse_html =True)
    folium.CircleMarker(
    [lat1,lng1],
    radius =5,
    popup = label,
    color ='blue',
    fill =True,
    fill_color='#3186cc',
    fill_opacity =0.7,
    parse_html =False).add_to(map_toronto)
map_toronto


# #### Let's find unique Borough in Toronto

# In[15]:


df_f['Borough'].unique()


# #### For simplification of the above map ,let's segment and cluster only the neighborhoods in Downtown Torronto. So let's slice the original dataframe and create a new dataframe of the Downtown Toronto data.

# In[16]:


downtown_dt = df_f[df_f['Borough']=='Downtown Toronto'].reset_index(drop =True)
downtown_dt.head()


# #### Let's find out the Latitude and Longitude of Dowtown

# In[17]:


address = 'Downtown Toronto, Ontario'

geolocator = Nominatim()
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Downtown Toronto are {}, {}.'.format(latitude, longitude))


# #### Now let's visualize Downtown Toronto's neighborhoods in it.

# In[18]:


downtown_map =folium.Map(location=[latitude,longitude],zoom_start =10)
for lat,lon,bor,nei in zip(downtown_dt['Latitude'],downtown_dt['Longitude'],downtown_dt['Borough'],downtown_dt['Neighbourhood']):
    label ='{},{}'.format(nei,bor)
    label = folium.Popup(label,parse_html =True)
    folium.CircleMarker(
    [lat,lon],
    radius =5,
    color ='blue',
    popup = label,
    fill =True,
    fill_color ='#3186cc',
    fill_opacity =0.7,    
    parse_html =False    
    ).add_to(downtown_map)
downtown_map    


# ### Next we will use 'Foursquare' API to explore the neighborhoods and segment them.
#   Defining Foursquare Credentials and Version

# In[19]:


CLIENT_ID = 'QTXQHVE25CP5XLGJ2N0IX4URBQTSCGJWYWTVWTIAQPF4KX42' 
CLIENT_SECRET = 'KLBUDY2YXVIZRGGWN2NPZ5BPP3NQHFA21P4EXZUAZPVK2RHN'
VERSION = '20180905'

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# ##### Let's explore the first neighborhood in our dataframe.
# the neighborhood's name & Get the neighborhood's latitude and longitude values:

# In[20]:


nei_lat = downtown_dt.loc[0,'Latitude']
nei_lon = downtown_dt.loc[0,'Longitude']
nei_name = downtown_dt.loc[0,'Neighbourhood']
print(' First Neighbour {} has latitude and longitude are {},{}'.format(nei_name,nei_lat,nei_lon))


# #### Now, let's get the top 100 venues that are in Marble Hill within a radius of 500 meters.

# In[21]:


LIMIT =100
radius =500
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    nei_lat, 
    nei_lon, 
    radius, 
    LIMIT)
url


# ##### Getting result by using GET -request method
# 

# In[22]:


result = requests.get(url).json()
#result


# ##### Now we are ready to clean the json and structure it into a pandas dataframe.

# In[24]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[25]:


ven = result['response']['groups'][0]['items']
nearby_ven = json_normalize(ven)
#print(nearby_ven)
filttered_col = ['venue.name','venue.categories','venue.location.lat','venue.location.lng']
nearby_ven = nearby_ven.loc[:,filttered_col]
#nearby_ven
nearby_ven['venue.categories']=nearby_ven.apply(get_category_type,axis=1)
nearby_ven.columns = [col.split(".")[-1] for col in nearby_ven.columns]
nearby_ven.head()


# ## 2. Explore Neighborhoods in Toronto

# ##### Now create a new dataframe called Downtown_venues for each Neighborhoods

# In[28]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[29]:


downtown_venues = getNearbyVenues(names = downtown_dt['Neighbourhood'],latitudes = downtown_dt['Latitude'],longitudes =downtown_dt['Longitude'])


# ##### Size of the new Data Frame

# In[30]:


print(downtown_venues.shape)
downtown_venues.head()


# #### Now let's check how many venues were returned for each neighborhood

# In[31]:


downtown_venues.groupby('Neighborhood').count().head()


# In[32]:


print('There are {} uniques categories.'.format(len(downtown_venues['Venue Category'].unique())))


# ## 3. Analyze Each Neighborhood

# In[33]:


downtown_category_dummies =pd.get_dummies(downtown_venues[['Venue Category']],prefix ="",prefix_sep="")
downtown_category_dummies['Neighborhood'] = downtown_venues['Neighborhood']
fixed_column = [downtown_category_dummies.columns[-1]] + list(downtown_category_dummies.columns[:-1])
downtown_category_dummies = downtown_category_dummies[fixed_column]
downtown_category_dummies.head()


# In[34]:


print('Size of the data_frame:',downtown_category_dummies.shape)


# #### Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[35]:


downtown_grouped = downtown_category_dummies.groupby('Neighborhood').mean().reset_index()
downtown_grouped.head()


# In[36]:


downtown_grouped.shape


# #### Each neighborhood along with the top 10 most common venues

# In[38]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[39]:


num_top_ven = 10
indicators = ['st','nd','rd']
column = ['Neighborhood']
for ind in np.arange(num_top_ven):
    try:
        column.append('{}{} Most Common Venue'.format(ind+1,indicators[ind]))
    except:
        column.append('{}th Most Common Venue'.format(ind+1))
neighborhood_venues_sort = pd.DataFrame(columns =column)
neighborhood_venues_sort['Neighborhood'] = downtown_grouped['Neighborhood']

for ind in np.arange(downtown_grouped.shape[0]):
    neighborhood_venues_sort.iloc[ind,1:] = return_most_common_venues(downtown_grouped.iloc[ind,:],num_top_ven)
neighborhood_venues_sort.head()


# ## 4. Cluster Neighborhoods

# In[40]:


kcl = 5
downtown_cl = downtown_grouped.drop('Neighborhood',axis =1)
kmeans =KMeans(n_clusters = kcl,random_state = 0).fit(downtown_cl)
kmeans.labels_


# #### Defining new dataframe that includes the cluster as well as the top 10 venues for each neighborhood

# In[41]:


downtown_merged = downtown_dt
downtown_merged['Cluster Labels'] = kmeans.labels_
downtown_merged.rename(columns ={'Neighbourhood':'Neighborhood'},inplace =True)

downtown_merged = downtown_merged.join(neighborhood_venues_sort.set_index('Neighborhood'),on = 'Neighborhood')

downtown_merged.head()


# #### visualizing the resulting clusters

# In[42]:


map_cluster = folium.Map(location =[latitude,longitude],zoom_start=11)
x = np.arange(kcl)
ys = [i+x+(i*x)**2 for i in range(kcl)]
color_arr = cm.rainbow(np.linspace(0,1,len(ys)))
rainbow = [colors.rgb2hex(i) for i in color_arr]

for lat,lng,poi,lab in zip(downtown_merged['Latitude'],downtown_merged['Longitude'],downtown_merged['Neighborhood'],downtown_merged['Cluster Labels']):
    labels = folium.Popup(str(poi)+' Cluster'+str(lab),parse_html =True)
    folium.CircleMarker(
    [lat,lng],
    radius =5,
    popup =labels,
    color = rainbow[lab-1],
    fill =True,
    fill_color = rainbow[lab-1],
    fill_opacity = 0.7,    
    ).add_to(map_cluster)
map_cluster 


# ## 5. Examine Clusters
# #### examining each cluster and determining the discriminating venue categories that distinguish each cluster
# #### Cluster1

# In[43]:


downtown_merged.loc[downtown_merged['Cluster Labels']==0,downtown_merged.columns[[2] + list(range(6,downtown_merged.shape[1]))]].head()


# #### Cluster2

# In[44]:


downtown_merged.loc[downtown_merged['Cluster Labels']==1,downtown_merged.columns[[2] + list(range(6,downtown_merged.shape[1]))]]


# #### Cluster3

# In[45]:


downtown_merged.loc[downtown_merged['Cluster Labels']==2,downtown_merged.columns[[2] + list(range(6,downtown_merged.shape[1]))]]


# #### Cluster4

# In[46]:


downtown_merged.loc[downtown_merged['Cluster Labels']==3,downtown_merged.columns[[2] + list(range(6,downtown_merged.shape[1]))]]


# #### Cluster5

# In[47]:


downtown_merged.loc[downtown_merged['Cluster Labels']==4,downtown_merged.columns[[2] + list(range(6,downtown_merged.shape[1]))]]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




