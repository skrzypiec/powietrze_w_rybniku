# coding: utf-8

#Libraries/functions import
from mpl_toolkits.basemap import Basemap

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from pylab import rcParams
rcParams['figure.figsize'] = 14, 10
import pyproj
import matplotlib.image as image
from lxml import etree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.collections import PatchCollection
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
from pysal.esda.mapclassify import Natural_Breaks as nb
from descartes import PolygonPatch
import fiona
from itertools import chain
from matplotlib import colors
import datetime
from pandas.io.json import json_normalize
import requests
import facebook
import matplotlib.ticker as plticker
from tabulate import tabulate
import credentials

from selenium import webdriver
import time
import pywinauto

# Convenience functions for working with colour ramps and bars
def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    """
    This is a convenience function to stop you making off-by-one errors
    Takes a standard colour ramp, and discretizes it,
    then draws a colour bar with correctly aligned labels
    """
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, **kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))
    if labels:
        colorbar.set_ticklabels(labels)
    return colorbar

def cmap_discretize(cmap, N):
    """
    Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)

    """
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in xrange(N + 1)]
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


# In[2]:


#Dates used in file names, maps etc.
current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
the_day = datetime.date.today() + datetime.timedelta(days=-1)
date_to_print = the_day.strftime("%d-%m-%Y")
date_to_file = the_day.strftime("%d%m%y")
current_time_to_file = datetime.datetime.now().strftime("%Y%m%d%H%M")


# In[3]:


headers = {
    'Accept': 'application/json',
    'apikey': credentials.airly,
}

params = (
    ('lat', '50.1'),
    ('lng', '18.5'),
    ('maxDistanceKM', '15'),
    ('maxResults', '50'))

response = requests.get('https://airapi.airly.eu/v2/installations/nearest', headers=headers, params=params)

#transform response to json
data_js = response.json()

df_temp = pd.DataFrame.from_records(data_js)

df_temp1 = df_temp['sponsor'].apply(pd.Series).drop(['id'], axis=1)
df_temp2 = df_temp['address'].apply(pd.Series)
df_temp3 = df_temp['location'].apply(pd.Series)

df_temp_all = pd.concat([df_temp, df_temp1, df_temp2, df_temp3], axis=1).query('name=="Rybnik" and airly==True')

sensors = df_temp_all[['id','displayAddress1','displayAddress2','latitude','longitude']].set_index('id')


#Fetch more detailed data, 24h history - one query per one sensor

ids = list(sensors.index)
ids.append(1127) #ids.append(822)(Widok) #Wroclaw
ids.append(820) #Krakow
ids.append(337) #Warszawa
ids.append(3432) #Gdansk (3401)(plac Wałowy)

headers = {
    'Accept': 'application/json',
    'apikey': credentials.airly,
}


allData = []
currentMeasure = []

for i, sensId in enumerate(ids):
    
    params = (
        ('installationId', str(sensId)),
    )
    

    response = requests.get('https://airapi.airly.eu/v2/measurements/installation', headers=headers, params=params)
    data_js = response.json()
    allData.append(pd.DataFrame.from_dict(json_normalize(data_js), orient='columns'))
    
    if(len(pd.DataFrame.from_dict(json_normalize(data_js), orient='columns')['current.values'][0])>1):
        currentMeasure.append(pd.pivot_table(json_normalize(allData[i]['current.values'].values[0]), columns='name', values='value'))
        currentMeasure[-1]['id'] = sensId


#Data about current air quality can be found in currentData df

currentData = pd.concat(currentMeasure)
currentData.set_index('id',inplace=True)

weather_graph = pd.DataFrame(currentData.query('id in @sensors.index').mean()).T


#Map creation

shp = fiona.open('data/city.shp')
bds = shp.bounds
shp.close()
extra = 0.01
ll = (bds[0], bds[1])
ur = (bds[2], bds[3])
coords = list(chain(ll, ur))
w, h = coords[2] - coords[0], coords[3] - coords[1]

m = Basemap(
    projection='tmerc',
    lon_0=18,
    lat_0=50.,
    ellps = 'WGS84',
    llcrnrlon=coords[0] - 2*extra * w,
    llcrnrlat=coords[1] - 2*extra + 0.01 * h,
    urcrnrlon=coords[2] + extra * w,
    urcrnrlat=coords[3] + extra + 0.01 * h,
    lat_ts=0,
    resolution='i',
    suppress_ticks=True)

m.readshapefile(
    'data/city',
    'rybnik',
    color='none',
    zorder=2)


# In[15]:


#Map dataframe set up
df_map = pd.DataFrame({
    'poly': [Polygon(xy) for xy in m.rybnik],
    'district': [ward['name'] for ward in m.rybnik_info],
    'id': [ward['id'] for ward in m.rybnik_info]})

df_map = df_map.sort_values('district')
df_map.set_index('id',inplace=True)

#It contains current data for every sensor
df_map = df_map.join(sensors, how='inner', lsuffix='sens').join(currentData, how='inner', rsuffix='current')
df_map = df_map.reset_index()


#Create Point objects in map coordinates from dataframe lon and lat values
map_points = pd.Series(
    [Point(m(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(sensors['longitude'], sensors['latitude'])])
plaque_points = MultiPoint(list(map_points.values))
wards_polygon = prep(MultiPolygon(list(df_map['poly'].values)))
ldn_points = filter(wards_polygon.contains, plaque_points)

df_map['count'] = df_map['poly'].map(lambda x: int(len(filter(prep(x).contains, ldn_points))))


# In[16]:


other_cities = currentData[['PM10']].query('id not in @sensors.index').reset_index()
other_cities['Miasto'] = other_cities.id.apply(lambda x: "Wrocław" if x==1127 else ("Kraków" if x==820 else ("Warszawa" if x==337 else ("Gdańsk" if x == 3432 else ""))))
other_cities.rename(index=str, columns={"PM10": "PM 10"}, inplace=True)
other_cities = other_cities.round(1).sort_values(by="PM 10", ascending = False)



#Create a map with current measurements for PM10 Index


pm10_labels = ['Bardzo niskie (0-25 $\mu$g/m$^3$)',
 'Niskie [25-50 $\mu$g/m$^3$)',
 'Średnie [50-90 $\mu$g/m$^3$)',
 'Wysokie [90-180 $\mu$g/m$^3$)',
    'Bardzo wysokie (>=180 $\mu$g/m$^3$)']

for i, text in enumerate(pm10_labels):
    pm10_labels[i] = text.decode('utf-8')
    
cmap = colors.ListedColormap(['#689f38','#8bc34a','#fbc02d','#f57c00','#ad1457'])
bounds= [0,25,50,90,180,1000]
norm = colors.BoundaryNorm(bounds, cmap.N)


plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, facecolor='w', frame_on=False)
fig.patch.set_facecolor('#f2f2f2')


# draw wards with black outlines
df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#000000', lw=2., alpha=1., zorder=4))
pc = PatchCollection(df_map['patches'], match_original=True)

pc.set_facecolor(cmap(norm(df_map['PM10'].values)))
ax.add_collection(pc)


# Add a colour bar
cb = colorbar_index(ncolors=len(pm10_labels), cmap=cmap, shrink=0.5, labels=pm10_labels)
cb.ax.tick_params(labelsize=14)


# Show highest densities, in descending order
highest = '\n'.join(
    value[1] + " - " + str(round(value['PM10'],1)) + " $\mu$g/m$^3$" for _, value in df_map.sort_values(by='PM10', ascending=False)[:5].iterrows())#.decode('utf-8')


highest = 'Najbardziej zanieczyszczone dzielnice:\n\n' + highest.decode('utf-8')
# Subtraction is necessary for precise y coordinate alignment
details = cb.ax.text(
    0., 1.55,
    highest,
    ha='left', va='top',
    size=14,
    color='#000000')

# Show current date-hour
date = ax.text(
    1.10, 0.033,
    current_time,
    ha='right', va='bottom',  weight = 'bold',
    size=16,
    color='#000000',
transform=ax.transAxes)


# # Show current temperature
tmpr = ax.text(
    1.2, 0.170,
    "Temperatura: %s $^{\circ}$C" %round(df_map['TEMPERATURE'].mean(),1),
    ha='left', va='bottom', 
    size=16,
    color='#000000',
transform=ax.transAxes)

# Show current pressure
prsr = ax.text(
    1.2, 0.130,
    u"Ciśnienie: %s hPa" %round(df_map['PRESSURE'].mean(),1),
    ha='left', va='bottom', 
    size=16,
    color='#000000',
transform=ax.transAxes)


# Bin method, copyright and source data info
smallprint = cb.ax.text(
    1.78, 0.028,
    '$\copyright$ Powietrze w Rybniku\n fb.com/powietrzewrybniku'.decode('utf-8'),
    ha='right', va='bottom',
    size=10,
    weight = 'bold',
    color='#000000',
    transform=ax.transAxes)


# Draw a map scale
m.drawmapscale(
    coords[0] + 0.06, coords[1] - 0.01,
    coords[0], coords[1], 
    8.,
    barstyle='fancy', labelstyle='simple',
    fillcolor1='w', fillcolor2='#000000',
    fontcolor='#000000', fontsize=12,
    zorder=5)

#Title
plt.title("Zanieczyszczenie PM$_{10}$".decode('utf-8'), fontsize=22, loc='left', weight='bold')


#Save the file
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.1)
fig.set_size_inches(12, 10)
plt.savefig('pictures/cur_pm10_%s.png'%current_time_to_file, dpi=200, alpha=True, bbox_inches='tight')
plt.show()


# # Generating message

message_other_cities = "Zanieczyszczenie PM 10, " + current_time +". W innych miastach Polski zanieczyszczenie wynosi: \n"
for index, row in other_cities.iterrows():
    message_other_cities = message_other_cities + row['Miasto'] + " - " +  str(row['PM 10']) + " ug/m3 (" +str(int(2*row['PM 10'])) + " %) \n"

message_other_cities = message_other_cities + "\nRybnik - dzielnica po dzielnicy: \n\n"    
message_other_cities = message_other_cities.decode('utf-8')

print_pm10 = df_map[['PM10','district']]
print_pm10['district'] = print_pm10['district'].apply(lambda x: x.decode('utf-8'))
print_pm10.rename(index=str, columns={"district": "Dzielnica", "PM10": "PM 10"}, inplace=True)
print_pm10.sort_values(by='PM 10', ascending=False, inplace=True)
print_pm10["PM 10"] = print_pm10["PM 10"].round(1).apply(lambda x: str(x) + " ug/m3 (" +str(int(2*x)) + " %)")
print_pm10.reset_index(drop=True, inplace=True)


message_ = tabulate(print_pm10.set_index("PM 10"), headers='keys',tablefmt= 'simple')

message_to_print = message_other_cities + message_



# # Message to be shown in a post

driver = webdriver.Chrome("C:/Users/Desktop/rybnik/chromedriver_win32/chromedriver.exe")

#go to webpage
driver.get("https://m.facebook.com/powietrzewrybniku")
time.sleep(5)

#login
elem = driver.find_element_by_id("mobile_login_bar")
elem.click()
time.sleep(5)

elem = driver.find_element_by_id("m_login_email")
elem.click()
elem.clear()
elem.send_keys(credentials.login)
time.sleep(1)

elem = driver.find_element_by_id("m_login_password")
elem.click()
elem.send_keys(credentials.password)
time.sleep(1)

elem = driver.find_element_by_name("login")
elem.click()
time.sleep(8)

#publishing part
try:
    elem= driver.find_element_by_xpath("//*[@id='action_bar']/div[2]")
    elem.click()

except:
    elem = driver.find_element_by_xpath("//*[@id='unit_id_1536492809980651']/div/div")
    elem.click()
    time.sleep(3)
    elem = driver.find_element_by_xpath("//*[@id='structured_composer_form']/div[5]/div/div[1]/button[1]")
    elem.click()

time.sleep(2)

#pywin part
pwa_app = pywinauto.application.Application().connect(title_re="Open")
time.sleep(3)

#upload photo with pm10 measurements
pwa_app.OpenDialog.Edit.SetText("C:\\Users\\Desktop\\rybnik\\pictures\\cur_pm10_%s.png"%current_time_to_file)
time.sleep(1)
pwa_app.OpenDialog.Button.click()
time.sleep(5)

elem = driver.find_element_by_name("status")
elem.click()
time.sleep(1)

#print message
elem.send_keys(message_to_print)
time.sleep(10)

#publish
elem = driver.find_element_by_xpath("//*[@id='composer-main-view-id']/div[1]/div/div[3]/div/button[1]")
elem.click()

#close
time.sleep(12)
driver.close()
