importScripts("https://cdn.jsdelivr.net/pyodide/v0.22.1/full/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/0.14.4/dist/wheels/bokeh-2.4.3-py3-none-any.whl', 'https://cdn.holoviz.org/panel/0.14.4/dist/wheels/panel-0.14.4-py3-none-any.whl', 'pyodide-http==0.1.0', 'folium', 'geopandas', 'holoviews>=1.15.4', 'hvplot', 'numpy', 'pandas', 'plotly']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  
import asyncio

from panel.io.pyodide import init_doc, write_doc

init_doc()

#!/usr/bin/env python
# coding: utf-8

# # Interactive Dashboard Python

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import panel as pn
import geopandas
import hvplot.pandas
from datetime import datetime
import folium
from panel.interact import interact
import plotly.express as px
import plotly.graph_objects as go
pn.extension()
pn.extension('plotly')
pn.extension('tabulator')


# In[2]:


url = 'https://raw.githubusercontent.com/Kefentse99/pubDas/main/finalDataset.csv'
df = pd.read_csv(url)


# In[3]:


#df = pd.read_csv('./finalDataset.csv')


# In[4]:


# Convert the 'Timestamp' column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract the hour from the timestamp for peak view times analysis
df['Hour'] = df['Timestamp'].dt.hour


# In[5]:


# Make DataFrame Pipeline Interactive
idf = df.interactive()


# ## world stats

# #### Most Visited Pages 

# In[6]:


most_visited = df['Requested URL'].value_counts()


# In[7]:


fig4 = px.bar(most_visited)
fig4.update_layout(xaxis_title='Visits', yaxis_title='Sports', title='Visits Per Sporting Code')
bar_chart  = pn.pane.Plotly(fig4)


# In[8]:


# Data preprocessing
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour

# Calculate the peak view times
peak_times = df['Hour'].value_counts().sort_index()

# Create a line plot using Plotly
fig6 = go.Figure(data=go.Scatter(x=peak_times.index, y=peak_times.values, mode='lines'))

# Set plot labels and title
fig6.update_layout(xaxis_title='Hour', yaxis_title='Visits', title='Peak View Times')

peak_times_chart = pn.pane.Plotly(fig6)


# ### Status Codes by Page

# In[9]:


# Compute value counts
value_counts = df['Response Code'].value_counts().reset_index()
value_counts.columns = ['Response Code', 'Count']


# In[10]:


# Create the pie chart
fig = px.pie(value_counts, values='Count', names='Response Code' , title='RESPONSE CODE BREAKDOWN')


# In[11]:


chart_panel = pn.pane.Plotly(fig)


# ### SPORTS CATEGORIES 

# In[12]:


df['Requested URL'].unique()


# In[13]:


# Define a dictionary to map page names to sports categories
sports_categories = {
    'marathons': 'athletics',
    'basketball': 'Ball Sports', 
    'football': 'Ball Sports',
    'hockey': 'Ball Sports',
    'baseball': 'Ball Sports',
    'volleyball': 'Ball Sports',
    'sprintraces': 'Athletics',
    'mid-distance-races': 'Athletics',
    'throwing_events': 'Athletics',
    'taekwondo': 'Martial Arts',
    'judo': 'Martial Arts',
    'searchsports': 'Miscellaneous',
    'jumping_events': 'Athletics',
    'gymnastics': 'Gymnastics',
    'index': 'Miscellaneous',
    'medalpoll': 'Miscellaneous'
}


# In[14]:


# Create a new column 'Sports Category' by mapping the page names
df['Sports Category'] = df['Requested URL'].map(sports_categories)


# In[15]:


mostCat = df['Sports Category'].value_counts()


# In[16]:


fig2 = px.bar(mostCat, orientation='h')
fig2.update_layout(xaxis_title='Visits', yaxis_title='Sports Category', title='Most Viewed Categories')

barCat = pn.pane.Plotly(fig2)


# In[17]:


#barCat = most_visited.hvplot.barh(xlabel='Sports Category', ylabel='Visits', title='Most Viewed Categories')


# In[18]:


df.head(3)


# ### Most viewed Per country 

# In[19]:


# renaming countries to merge with world data
df['CountryName']=np.where(df['CountryName']=='Russian Federation','Russia',df['CountryName'])
df['CountryName']=np.where(df['CountryName']=='Republic of Korea','South Korea',df['CountryName'])
df['CountryName']=np.where(df['CountryName']=='United States','United States of America',df['CountryName'])
df['CountryName']=np.where(df['CountryName']=='Dominica','Dominican Rep.',df['CountryName'])


# In[20]:


# Group the data by 'Country' and 'RequestedPage', and count the occurrences
highCountry = df.groupby(['CountryName', 'Requested URL']).size().reset_index(name='Count2')


# In[21]:


# Sort the data by count in descending order within each country group
highCountry_sorted = highCountry.sort_values(['CountryName' , 'Count2'], ascending=[True, False])


# In[22]:


# Get the most visited page for each country
pageCountry = highCountry_sorted.groupby('CountryName').first()


# In[23]:


pageCountry.head()


# ## country level view

# In[24]:


### reading country level data from geopandas for country boundaries
world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
print("Geometry Column Name : ", world.geometry.name)
print("Dataset Size : ", world.shape)
world.head()


# In[25]:


##checking country name not present in world data
set(df['CountryName'].unique())-set(world['name'].unique())


# In[26]:


# renaming countries to merge with world data
df['CountryName']=np.where(df['CountryName']=='Russian Federation','Russia',df['CountryName'])
df['CountryName']=np.where(df['CountryName']=='Republic of Korea','South Korea',df['CountryName'])
df['CountryName']=np.where(df['CountryName']=='United States','United States of America',df['CountryName'])
df['CountryName']=np.where(df['CountryName']=='Dominica','Dominican Rep.',df['CountryName'])


# In[27]:


# Calculate the sum of country counts
country_counts = df['CountryName'].value_counts()


# In[28]:


country_counts.head()


# In[29]:


# Merge the country counts with the geospatial data
world = world.merge(country_counts, how='left', left_on='name', right_index=True)
world['Count'] = world['CountryName'].fillna(0).astype(int)


# In[30]:


world = world.merge(pageCountry, how='left', left_on='name', right_index=True)


# In[31]:


world.head()


# In[32]:


## plotting map with folium
urban_area_map = folium.Map( location=[0, 0],zoom_start=2.2)
title_html = '''
             <h3 align="center" style="font-size:20px"><b>Viewership By Country </b></h3>
             <h6 align="center" style="font-size:14px"><b>Yokyo 2023 Fun Olympic  </b></h6>
             '''
urban_area_map.get_root().html.add_child(folium.Element(title_html))
folium.Choropleth(
    geo_data=world,
    name='choropleth',
    data=world,
    columns=['name', 'Count', 'Requested URL'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    nan_fill_color='White',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Viewership Per Country'
).add_to(urban_area_map)


# In[33]:


style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}


# In[34]:


NIL = folium.features.GeoJson(
    world,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['name', 'Count', 'Requested URL'],
        aliases=['Country: ','Visits' , 'Most Visited Page'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 5px;") 
    )
)
urban_area_map.add_child(NIL)
urban_area_map.keep_in_front(NIL)
folium.LayerControl().add_to(urban_area_map)
urban_area_map


# In[35]:


pn.Column(
    pn.Row(pn.Column(pn.Row(bar_chart),pn.Row(peak_times_chart),pn.Row(barCat))),
    pn.Spacer(width=50),
    pn.Row(pn.panel(urban_area_map,height=600)),

)


# ### creating dashboard

# In[39]:


#Layout using Template
template = pn.template.FastListTemplate(
    title='Yokyo Fun Olympics Website analytics', 
    sidebar=[pn.pane.Markdown("# Yokyo Fun Olympics "), 
             pn.pane.Markdown("####This complied data is sourced from the weblog files of the Yokyo Fun Olympic website"), 
           #  pn.pane.PNG("olympics.png", sizing_mode='scale_both'),
            ],
    main=[
    pn.Row(pn.Column(bar_chart, barCat),pn.Column(chart_panel , peak_times_chart),background='White'),
    #pn.Spacer(width=50),
    pn.Row(pn.panel(urban_area_map,height=600),background='White')
   
    ],
    accent_base_color="#1e1757",
    header_background="#2731BE",
)


# In[40]:


#template.show()
template.servable();


# In[ ]:






await write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.runPythonAsync(`
    import json

    state.curdoc.apply_json_patch(json.loads('${msg.patch}'), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads("""${msg.location}""")
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()