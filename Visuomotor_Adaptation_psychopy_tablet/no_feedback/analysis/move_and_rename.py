#%% imports
import os
import shutil
# %%
source = 'C:\\Users\\Tejas\\Documents\\Lab-Codes\\Visuomotor_Adaptation_psychopy_tablet\\no_feedback\\data'
dest = 'C:\\Users\\Tejas\\Documents\\Lab-Codes\\Visuomotor_Adaptation_psychopy_tablet\\no_feedback\\analysis\\data'
# %%
files = os.listdir(source)
if not os.path.exists(dest):
    os.makedirs(dest)
# %%
os.chdir(source)
for f in files:
    if(f.endswith('.csv')):
        shutil.move(f, dest)
# %%
os.chdir(dest)
files = os.listdir(dest)
count = 0
for f in files:
    if(f.startswith('20')):
        os.rename(f, f[:4]+'.csv')
    #count = count + 1

# %%
