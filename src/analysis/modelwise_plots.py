#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns


#%%
df = pd.read_csv('results_micro.csv')
df.head()
#%%
models = df['model'].unique()
print(list(models))
#%% set all llama models to shades of purple
colors = []
shade_purple = np.linspace(0.1, 0.9, 5)
shade_blue = np.linspace(0.1, 0.9, 5)
shade_red = np.linspace(0.1, 0.9, 5)
shade_green = np.linspace(0.1, 0.9, 5)

colors_7b = [(x/5.0, x/8.0, 0.75) for x in range(4)[::-1]] # <-- Quick gradient example along the Red/Green dimensions.
colors_13b = [(0.7, x/5.0, x/8.0) for x in range(4)[::-1]] # <-- Quick gradient example along the Blue dimension.
colors_30b = [(x/5.0, 0.7, x/8.0) for x in range(4)[::-1]] # <-- Quick gradient example along the Red dimension.

colors_llama2 = [(0.5, 0.5, x/5.0) for x in range(3)[::-1]] # <-- Quick gradient example along the Blue dimension.

colors_olmo_7b = [(x/4.0, x/9.0, 0.85) for x in range(2)[::-1]] # <-- Quick gradient example along the Red/Green dimensions.


color_mistral = (0.5, 0.2, 0.5)
color_gpt4 = (0.5, 0.5, 0.3)
color_gpt3 = (0.9, 0.5, 0.5)

colors = colors_7b + colors_13b + colors_30b + colors_llama2 + colors_olmo_7b + [color_mistral, color_gpt4, color_gpt3]

#%%
df_country = df[df['type'] == 'country_conditioned']['accuracy']

np_country = df_country.to_numpy()
# append 0.33 to the beginning of the array
#np_country = np.insert(np_country, 0, 0.33)
# all_models = ['baseline reference'] + list(models)
# colors = [(0.5, 0.5, 0.5)] + colors

#%%
fig, ax = plt.subplots()
# xvals = [1, 3,4,5,6 , 8,9,10,11, 13,14,15,16, 18,19,20, 22,23, 25, 27, 29]
xvals = [1,2,3,4 , 6,7,8,9, 11,12,13,14, 16,17,18, 20,21, 23, 25, 27]

# set figsize to be very wide
fig.set_size_inches(15, 5)
# multiply xvals by 3 and increase width
xvals = [x*3 for x in xvals]
width = 2.5
# set fontsize
plt.rcParams.update({'font.size': 15})


ax.bar(xvals, np_country, width=width, color=colors)
# set model name as xticks
ax.set_xticks(xvals)
ax.set_xticklabels(models, rotation=60)
# set yaxis between 0 and 1
ax.set_ylim([0, 1])

# set rounded bars and gridlines
ax.yaxis.grid(True)
# set grid for every 0.1
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
# send the grid to back
ax.set_axisbelow(True)


# add acc values on top of bars
for i, v in enumerate(np_country):
    ax.text(xvals[i]-1.25, v + 0.03, str(round(v, 2)), color='black', fontsize=12, fontweight=525)
# add a horizontal dashed line at 0.33
plt.axhline(y=0.33, color='teal', linestyle='--', linewidth=2.5, label='baseline reference')
# set legend
plt.legend(loc='upper right', fontsize=15)

# set title to be country conditioned
ax.set_title('Country Conditioned accuracy for all models')
plt.savefig('country_conditioned.png')

# repeat for value and rot
#%%
df_value = df[df['type'] == 'value_conditioned']['accuracy']
np_value = df_value.to_numpy()

fig, ax = plt.subplots()
# xvals = [1, 3,4,5,6 , 8,9,10,11, 13,14,15,16, 18,19,20, 22,23, 25, 27, 29]
xvals = [1,2,3,4 , 6,7,8,9, 11,12,13,14, 16,17,18, 20,21, 23, 25, 27]

# set figsize to be very wide
fig.set_size_inches(15, 5)
# multiply xvals by 3 and increase width
xvals = [x*3 for x in xvals]
width = 2.5
# set fontsize
plt.rcParams.update({'font.size': 15})


ax.bar(xvals, np_value, width=width, color=colors)
# set model name as xticks
ax.set_xticks(xvals)
ax.set_xticklabels(models, rotation=60)
# set yaxis between 0 and 1
ax.set_ylim([0, 1])

# set rounded bars and gridlines
ax.yaxis.grid(True)
# set grid for every 0.1
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
# send the grid to back
ax.set_axisbelow(True)


# add acc values on top of bars
for i, v in enumerate(np_value):
    ax.text(xvals[i]-1.25, v + 0.03, str(round(v, 2)), color='black', fontsize=12, fontweight=525)
# add a horizontal dashed line at 0.33
plt.axhline(y=0.33, color='teal', linestyle='--', linewidth=2.5, label='baseline reference')
# set legend
plt.legend(loc='upper right', fontsize=15)

# set title to be value conditioned
ax.set_title('Value Conditioned accuracy for all models')
plt.savefig('value_conditioned.png')
#%%
df_rot = df[df['type'] == 'rot_conditioned']['accuracy']
np_rot = df_rot.to_numpy()

fig, ax = plt.subplots()
# xvals = [1, 3,4,5,6 , 8,9,10,11, 13,14,15,16, 18,19,20, 22,23, 25, 27, 29]
xvals = [1,2,3,4 , 6,7,8,9, 11,12,13,14, 16,17,18, 20,21, 23, 25, 27]

# set figsize to be very wide
fig.set_size_inches(15, 5)
# multiply xvals by 3 and increase width
xvals = [x*3 for x in xvals]
width = 2.5
# set fontsize
plt.rcParams.update({'font.size': 15})


ax.bar(xvals, np_rot, width=width, color=colors)
# set model name as xticks
ax.set_xticks(xvals)
ax.set_xticklabels(models, rotation=60)
# set yaxis between 0 and 1
ax.set_ylim([0, 1])

# set rounded bars and gridlines
ax.yaxis.grid(True)
# set grid for every 0.1
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
# send the grid to back
ax.set_axisbelow(True)


# add acc values on top of bars
for i, v in enumerate(np_rot):
    ax.text(xvals[i]-1.25, v + 0.03, str(round(v, 2)), color='black', fontsize=12, fontweight=525)
# add a horizontal dashed line at 0.33
plt.axhline(y=0.33, color='teal', linestyle='--', linewidth=2.5, label='baseline reference')
# set legend
plt.legend(loc='upper right', fontsize=15)

# set title to be rot conditioned
ax.set_title('Rule-of-thumb Conditioned accuracy for all models')
plt.savefig('rot_conditioned.png')
#%%
