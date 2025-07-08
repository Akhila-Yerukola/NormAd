#%%
import pandas as pd

df = pd.read_csv('../output/Etiquette_gpt4_3opt_final.csv')
map_iw_bin_to_country = {
    'African-Islamic': ['afghanistan', 'bangladesh','egypt','ethiopia','iran','iraq', 'kenya', 'lebanon', 'mauritius', 'pakistan','palestinian_territories','saudi_arabia', 'somalia', 'south_africa','south_sudan','sudan','syria', 'türkiye','zimbabwe'], # mauritius 
    'Orthodox Europe': ['bosnia_and_herzegovina','greece','north_macedonia','romania','russia', 'serbia','ukraine'],
    'Confucian': ['china','hong_kong','japan','south_korea','taiwan'],
    'Catholic Europe': ['austria','croatia','hungary','italy','poland','portugal','spain'],
    'Protestant Europe': ['france','germany','netherlands','sweden', 'malta'], # malta
    'English Speaking': ['australia','canada','ireland','new_zealand','united_kingdom','united_states_of_america'],
    'Latin America': ['argentina','brazil','chile','colombia','mexico','peru','venezuela'], 
    'West and South Asia': ['cambodia','cyprus','fiji','india','indonesia','israel','laos','malaysia','myanmar','nepal','papua_new_guinea','samoa','singapore','sri_lanka','thailand', 'timor-leste', 'tonga', 'vietnam','philippines'] # philippines, papua new guinea, samoa, timor-leste, tonga
}
#%%
df['iw_bin'] = df['Country'].apply(lambda x: [k for k,v in map_iw_bin_to_country.items() if x in v][0])
grouped = df.groupby(by=['Gold Label', 'iw_bin']).count()['Country'].reset_index()

# Pivot the table to set iw_bin as columns and Gold Label as rows with counts as elements
pivot_table = grouped.pivot_table(index='Gold Label', columns='iw_bin', values='Country', aggfunc='sum')

# Fill NaN values with 0 if needed
pivot_table = pivot_table.fillna(0)

print(pivot_table.to_latex())