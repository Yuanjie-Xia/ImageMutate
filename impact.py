import pandas as pd
import scipy.stats as stats
df = pd.read_csv('djpeg_reverse_performance.csv',index_col=False)
result = stats.ks_2samp(df.iloc[:,1],df.iloc[:,2])
print(result)
