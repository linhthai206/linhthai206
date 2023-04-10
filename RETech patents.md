# Assignment 4: Merging

This assignment requires you to 
- Explore some new data on patents
- Implement merges according to [best practices](https://ledatascifi.github.io/ledatascifi-2023/content/03/05b_merging.html#merging-in-new-variables-to-your-analysis)
- I did not include questions explicitly aimed at [Merging in new variables to your analysis](https://ledatascifi.github.io/ledatascifi-2023/content/03/05b_merging.html#merging-in-new-variables-to-your-analysis) or [Create your variables before a merge when possible](https://ledatascifi.github.io/ledatascifi-2023/content/03/05b_merging.html#create-your-variables-before-a-merge-when-possible), but **you should read those** because they will matter for the midterm project and your group projects a lot!


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import seaborn as sns

# these three are used to open the CCM dataset:
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
```

## Download CCM data

This code comes from the textbook.

To get the URL, I [went to the data repo and found it](https://github.com/LeDataSciFi/data/blob/main/Firm%20Year%20Datasets%20(Compustat)/CCM_cleaned_for_class.zip), and then I right clicked on the "Download" button and copied the link. 


```python
url = 'https://github.com/LeDataSciFi/data/raw/main/Firm%20Year%20Datasets%20(Compustat)/CCM_cleaned_for_class.zip'

#ccm = pd.read_stata(url)   
# <-- that code would work if I had uploaded the data as a csv file, but GH said it was too big to upload 
# so I zipped it. We need a little workaround to download it:

with urlopen(url) as request:
    data = BytesIO(request.read())

with ZipFile(data) as archive:
    with archive.open(archive.namelist()[0]) as stata:
        ccm = pd.read_stata(stata)

ccm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gvkey</th>
      <th>fyear</th>
      <th>datadate</th>
      <th>lpermno</th>
      <th>gsector</th>
      <th>sic</th>
      <th>sic3</th>
      <th>age</th>
      <th>tic</th>
      <th>state</th>
      <th>...</th>
      <th>tnic3hhi</th>
      <th>tnic3tsimm</th>
      <th>prodmktfluid</th>
      <th>delaycon</th>
      <th>equitydelaycon</th>
      <th>debtdelaycon</th>
      <th>privdelaycon</th>
      <th>l_emp</th>
      <th>l_ppent</th>
      <th>l_laborratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>1975.0</td>
      <td>1975-12-31</td>
      <td>25881.0</td>
      <td></td>
      <td>3089.0</td>
      <td>308.0</td>
      <td>0.0</td>
      <td>AE.2</td>
      <td></td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.719789</td>
      <td>2.111788</td>
      <td>1.930200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000</td>
      <td>1976.0</td>
      <td>1976-12-31</td>
      <td>25881.0</td>
      <td></td>
      <td>3089.0</td>
      <td>308.0</td>
      <td>1.0</td>
      <td>AE.2</td>
      <td></td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.900161</td>
      <td>2.858766</td>
      <td>2.421281</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>1977.0</td>
      <td>1977-12-31</td>
      <td>25881.0</td>
      <td></td>
      <td>3089.0</td>
      <td>308.0</td>
      <td>2.0</td>
      <td>AE.2</td>
      <td></td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.008323</td>
      <td>3.040562</td>
      <td>2.437114</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1001</td>
      <td>1983.0</td>
      <td>1983-12-31</td>
      <td>10015.0</td>
      <td>25</td>
      <td>5812.0</td>
      <td>581.0</td>
      <td>0.0</td>
      <td>AMFD.</td>
      <td>OK</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.869618</td>
      <td>2.255074</td>
      <td>1.817871</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1001</td>
      <td>1984.0</td>
      <td>1984-12-31</td>
      <td>10015.0</td>
      <td>25</td>
      <td>5812.0</td>
      <td>581.0</td>
      <td>1.0</td>
      <td>AMFD.</td>
      <td>OK</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.917090</td>
      <td>2.618490</td>
      <td>2.135985</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>222996</th>
      <td>297209</td>
      <td>2012.0</td>
      <td>2012-12-31</td>
      <td>13104.0</td>
      <td>10</td>
      <td>1381.0</td>
      <td>138.0</td>
      <td>1.0</td>
      <td>PACD</td>
      <td></td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.471877</td>
      <td>8.232553</td>
      <td>8.738124</td>
    </tr>
    <tr>
      <th>222997</th>
      <td>297209</td>
      <td>2013.0</td>
      <td>2013-12-31</td>
      <td>13104.0</td>
      <td>10</td>
      <td>1381.0</td>
      <td>138.0</td>
      <td>2.0</td>
      <td>PACD</td>
      <td></td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.663718</td>
      <td>8.414751</td>
      <td>8.474280</td>
    </tr>
    <tr>
      <th>222998</th>
      <td>311524</td>
      <td>2013.0</td>
      <td>2013-12-31</td>
      <td>13861.0</td>
      <td>15</td>
      <td>2860.0</td>
      <td>286.0</td>
      <td>0.0</td>
      <td>TAM</td>
      <td>PA</td>
      <td>...</td>
      <td>0.326118</td>
      <td>1.1371</td>
      <td>5.650078</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.615726</td>
      <td>6.154858</td>
      <td>6.314076</td>
    </tr>
    <tr>
      <th>222999</th>
      <td>315887</td>
      <td>2013.0</td>
      <td>2013-12-31</td>
      <td>14344.0</td>
      <td>20</td>
      <td>4412.0</td>
      <td>441.0</td>
      <td>0.0</td>
      <td>SALT</td>
      <td></td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.001000</td>
      <td>5.920753</td>
      <td>12.825821</td>
    </tr>
    <tr>
      <th>223000</th>
      <td>316056</td>
      <td>2013.0</td>
      <td>2013-12-31</td>
      <td>14297.0</td>
      <td>20</td>
      <td>3420.0</td>
      <td>342.0</td>
      <td>0.0</td>
      <td>ALLE</td>
      <td></td>
      <td>...</td>
      <td>0.554508</td>
      <td>1.0259</td>
      <td>1.421064</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.197225</td>
      <td>5.318120</td>
      <td>3.233764</td>
    </tr>
  </tbody>
</table>
<p>223001 rows × 43 columns</p>
</div>




```python
url1 = 'https://github.com/LeDataSciFi/data/raw/main/Firm%20Year%20Datasets%20(Compustat)/firmyear_patstats.csv'

firmyear_patstats = pd.read_csv(url1)
firmyear_patstats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gvkey</th>
      <th>ayear</th>
      <th>patent_app_count</th>
      <th>RETech_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>1974</td>
      <td>2</td>
      <td>1.282584</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000</td>
      <td>1975</td>
      <td>2</td>
      <td>1.309539</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>1976</td>
      <td>2</td>
      <td>1.099830</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1004</td>
      <td>1979</td>
      <td>1</td>
      <td>0.462650</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1004</td>
      <td>1981</td>
      <td>1</td>
      <td>2.276393</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>86375</th>
      <td>311524</td>
      <td>2009</td>
      <td>1</td>
      <td>4.002895</td>
    </tr>
    <tr>
      <th>86376</th>
      <td>318728</td>
      <td>2015</td>
      <td>1</td>
      <td>1.407193</td>
    </tr>
    <tr>
      <th>86377</th>
      <td>318728</td>
      <td>2016</td>
      <td>2</td>
      <td>1.995176</td>
    </tr>
    <tr>
      <th>86378</th>
      <td>332115</td>
      <td>2000</td>
      <td>1</td>
      <td>1.765147</td>
    </tr>
    <tr>
      <th>86379</th>
      <td>349530</td>
      <td>2017</td>
      <td>1</td>
      <td>0.521259</td>
    </tr>
  </tbody>
</table>
<p>86380 rows × 4 columns</p>
</div>



## Firm year patent data.

We will use `firmyear_patstats.csv`. It's in the same place in the data repo as the CCM file. 

It contains variables that we want to include in the  CCM dataset for analysis.

## PART 1

Insert cell(s) below this one as needed to finish this Part.

Load the following two datasets and answer these questions. Assume for these questions that the `ccm` data is the "left" dataset and the `firmyear_patstats` is the "right" dataset. 

1. How many observations are there in `ccm` data?


```python
len(ccm)
```




    223001



2. How many observations are there in `firmyear_patstats` data?


```python
len(firmyear_patstats)
```




    86380



3. After an inner merge?


```python
# change data type for ccm['fyear'] into int64
ccm['fyear'] = ccm['fyear'].astype(np.int64)
```


```python
len(
    ccm.merge(firmyear_patstats.rename(columns={'ayear':'fyear'}),
          how='inner',
          on=['gvkey','fyear'],
          indicator=True,
          validate='1:1')
)
```




    49130



4. How many observations are there after a left merge?


```python
len(
    ccm.merge(firmyear_patstats.rename(columns={'ayear':'fyear'}),
          how='left',
          on=['gvkey','fyear'],
          indicator=True,
          validate='1:1')
)
```




    223001



5. After a right merge? 


```python
len(
    ccm.merge(firmyear_patstats.rename(columns={'ayear':'fyear'}),
          how='right',
          on=['gvkey','fyear'],
          indicator=True,
          validate='1:1')
)
```




    86380



6. After an outer merge? 


```python
len(
    ccm.merge(firmyear_patstats.rename(columns={'ayear':'fyear'}),
          how='outer',
          on=['gvkey','fyear'],
          indicator=True,
          validate='1:1')
)
```




    260251



7. Why isn't the answer to Q4 and Q5 the same?

The difference lies in the nature of the two merging methods. Left merge contains 'inner' observations and all unmatched observations in the left dataframe (ccm), thus retaining the initial number of observations in ccm. While right merge contains 'inner' observations while keeping the unmatched observations in the right dataframe (firmyear_patstats), totaling the initial number of observations in firmyear_patstats.

8. Is this a 1:1, 1:M, M:1, or M:M merge?

1:1 merge because when we are merging on both gvkey and fyear as identifiers, there is only one match from the left and right dataframe. There should be only one observation for one gvkey in one given year provided the number of patent applications

Remember: Specify `how`, `on`, `indicator=True`, and `validate` on each merge!

## Part 2: Industry patenting trends

- Reduce the data to gsectors 15, 35, 20, 45, 40, 25, and years 1980-2010.
- Calculate the average RETech of each **patent** each industry each year
    - **Be thoughtful about how you compute this.** There is a correct answer. Don't ask questions about this on the discussion board. 


```python
firm_year_patstats = ccm.merge(firmyear_patstats.rename(columns={'ayear':'fyear'}),
          how='inner',
          on=['gvkey','fyear'],
          indicator=True,
          validate='1:1')
firm_year_patstats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gvkey</th>
      <th>fyear</th>
      <th>datadate</th>
      <th>lpermno</th>
      <th>gsector</th>
      <th>sic</th>
      <th>sic3</th>
      <th>age</th>
      <th>tic</th>
      <th>state</th>
      <th>...</th>
      <th>delaycon</th>
      <th>equitydelaycon</th>
      <th>debtdelaycon</th>
      <th>privdelaycon</th>
      <th>l_emp</th>
      <th>l_ppent</th>
      <th>l_laborratio</th>
      <th>patent_app_count</th>
      <th>RETech_avg</th>
      <th>_merge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>1975</td>
      <td>1975-12-31</td>
      <td>25881.0</td>
      <td></td>
      <td>3089.0</td>
      <td>308.0</td>
      <td>0.0</td>
      <td>AE.2</td>
      <td></td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.719789</td>
      <td>2.111788</td>
      <td>1.930200</td>
      <td>2</td>
      <td>1.309539</td>
      <td>both</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000</td>
      <td>1976</td>
      <td>1976-12-31</td>
      <td>25881.0</td>
      <td></td>
      <td>3089.0</td>
      <td>308.0</td>
      <td>1.0</td>
      <td>AE.2</td>
      <td></td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.900161</td>
      <td>2.858766</td>
      <td>2.421281</td>
      <td>2</td>
      <td>1.099830</td>
      <td>both</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1004</td>
      <td>1979</td>
      <td>1980-05-31</td>
      <td>54594.0</td>
      <td>20</td>
      <td>5080.0</td>
      <td>508.0</td>
      <td>4.0</td>
      <td>AIR</td>
      <td>IL</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.639746</td>
      <td>2.677454</td>
      <td>2.716054</td>
      <td>1</td>
      <td>0.462650</td>
      <td>both</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1004</td>
      <td>1981</td>
      <td>1982-05-31</td>
      <td>54594.0</td>
      <td>20</td>
      <td>5080.0</td>
      <td>508.0</td>
      <td>6.0</td>
      <td>AIR</td>
      <td>IL</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.900161</td>
      <td>3.512470</td>
      <td>3.103757</td>
      <td>1</td>
      <td>2.276393</td>
      <td>both</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1004</td>
      <td>1982</td>
      <td>1983-05-31</td>
      <td>54594.0</td>
      <td>20</td>
      <td>5080.0</td>
      <td>508.0</td>
      <td>7.0</td>
      <td>AIR</td>
      <td>IL</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.841567</td>
      <td>3.494870</td>
      <td>3.186415</td>
      <td>1</td>
      <td>-0.265829</td>
      <td>both</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49125</th>
      <td>287882</td>
      <td>2011</td>
      <td>2011-12-31</td>
      <td>92793.0</td>
      <td>10</td>
      <td>2911.0</td>
      <td>291.0</td>
      <td>3.0</td>
      <td>EC</td>
      <td></td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.275111</td>
      <td>10.067163</td>
      <td>7.900469</td>
      <td>1</td>
      <td>2.070726</td>
      <td>both</td>
    </tr>
    <tr>
      <th>49126</th>
      <td>289735</td>
      <td>2013</td>
      <td>2013-12-31</td>
      <td>14060.0</td>
      <td>45</td>
      <td>4899.0</td>
      <td>489.0</td>
      <td>0.0</td>
      <td>XGTI</td>
      <td>FL</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.079735</td>
      <td>0.591114</td>
      <td>2.273243</td>
      <td>4</td>
      <td>0.706199</td>
      <td>both</td>
    </tr>
    <tr>
      <th>49127</th>
      <td>293754</td>
      <td>2011</td>
      <td>2011-12-31</td>
      <td>12667.0</td>
      <td>35</td>
      <td>2836.0</td>
      <td>283.0</td>
      <td>0.0</td>
      <td>MDGN</td>
      <td>PA</td>
      <td>...</td>
      <td>-0.050327</td>
      <td>0.165006</td>
      <td>-0.039408</td>
      <td>0.175057</td>
      <td>0.030529</td>
      <td>0.360468</td>
      <td>2.639057</td>
      <td>1</td>
      <td>0.515078</td>
      <td>both</td>
    </tr>
    <tr>
      <th>49128</th>
      <td>295786</td>
      <td>2013</td>
      <td>2013-12-31</td>
      <td>14144.0</td>
      <td>20</td>
      <td>3523.0</td>
      <td>352.0</td>
      <td>0.0</td>
      <td>CNHI</td>
      <td></td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.279329</td>
      <td>9.005774</td>
      <td>4.740270</td>
      <td>179</td>
      <td>0.711167</td>
      <td>both</td>
    </tr>
    <tr>
      <th>49129</th>
      <td>296885</td>
      <td>2012</td>
      <td>2012-12-31</td>
      <td>13707.0</td>
      <td>35</td>
      <td>2836.0</td>
      <td>283.0</td>
      <td>0.0</td>
      <td>RDHL</td>
      <td></td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.107059</td>
      <td>NaN</td>
      <td>2</td>
      <td>1.608207</td>
      <td>both</td>
    </tr>
  </tbody>
</table>
<p>49130 rows × 46 columns</p>
</div>




```python
subsample = firm_year_patstats.query("(gsector == ['15', '35', '20', '45', '40', '25']) & (1980 <= fyear <= 2010)")
subsample
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gvkey</th>
      <th>fyear</th>
      <th>datadate</th>
      <th>lpermno</th>
      <th>gsector</th>
      <th>sic</th>
      <th>sic3</th>
      <th>age</th>
      <th>tic</th>
      <th>state</th>
      <th>...</th>
      <th>delaycon</th>
      <th>equitydelaycon</th>
      <th>debtdelaycon</th>
      <th>privdelaycon</th>
      <th>l_emp</th>
      <th>l_ppent</th>
      <th>l_laborratio</th>
      <th>patent_app_count</th>
      <th>RETech_avg</th>
      <th>_merge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>1004</td>
      <td>1981</td>
      <td>1982-05-31</td>
      <td>54594.0</td>
      <td>20</td>
      <td>5080.0</td>
      <td>508.0</td>
      <td>6.0</td>
      <td>AIR</td>
      <td>IL</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.900161</td>
      <td>3.512470</td>
      <td>3.103757</td>
      <td>1</td>
      <td>2.276393</td>
      <td>both</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1004</td>
      <td>1982</td>
      <td>1983-05-31</td>
      <td>54594.0</td>
      <td>20</td>
      <td>5080.0</td>
      <td>508.0</td>
      <td>7.0</td>
      <td>AIR</td>
      <td>IL</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.841567</td>
      <td>3.494870</td>
      <td>3.186415</td>
      <td>1</td>
      <td>-0.265829</td>
      <td>both</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1004</td>
      <td>1985</td>
      <td>1986-05-31</td>
      <td>54594.0</td>
      <td>20</td>
      <td>5080.0</td>
      <td>508.0</td>
      <td>10.0</td>
      <td>AIR</td>
      <td>IL</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.022451</td>
      <td>3.679183</td>
      <td>3.077002</td>
      <td>2</td>
      <td>1.035338</td>
      <td>both</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1004</td>
      <td>1988</td>
      <td>1989-05-31</td>
      <td>54594.0</td>
      <td>20</td>
      <td>5080.0</td>
      <td>508.0</td>
      <td>13.0</td>
      <td>AIR</td>
      <td>IL</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.264127</td>
      <td>3.946811</td>
      <td>2.995142</td>
      <td>1</td>
      <td>3.164818</td>
      <td>both</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1004</td>
      <td>1989</td>
      <td>1990-05-31</td>
      <td>54594.0</td>
      <td>20</td>
      <td>5080.0</td>
      <td>508.0</td>
      <td>14.0</td>
      <td>AIR</td>
      <td>IL</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.232560</td>
      <td>4.165750</td>
      <td>3.262219</td>
      <td>1</td>
      <td>-0.576573</td>
      <td>both</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49101</th>
      <td>270705</td>
      <td>2008</td>
      <td>2008-12-31</td>
      <td>91163.0</td>
      <td>45</td>
      <td>3674.0</td>
      <td>367.0</td>
      <td>2.0</td>
      <td>HIMX</td>
      <td></td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.794801</td>
      <td>4.027332</td>
      <td>3.815429</td>
      <td>136</td>
      <td>0.660242</td>
      <td>both</td>
    </tr>
    <tr>
      <th>49102</th>
      <td>270705</td>
      <td>2009</td>
      <td>2009-12-31</td>
      <td>91163.0</td>
      <td>45</td>
      <td>3674.0</td>
      <td>367.0</td>
      <td>3.0</td>
      <td>HIMX</td>
      <td></td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.801553</td>
      <td>3.962450</td>
      <td>3.737050</td>
      <td>136</td>
      <td>0.017882</td>
      <td>both</td>
    </tr>
    <tr>
      <th>49103</th>
      <td>270705</td>
      <td>2010</td>
      <td>2010-12-31</td>
      <td>91163.0</td>
      <td>45</td>
      <td>3674.0</td>
      <td>367.0</td>
      <td>4.0</td>
      <td>HIMX</td>
      <td></td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.850578</td>
      <td>3.882821</td>
      <td>3.568598</td>
      <td>106</td>
      <td>0.517935</td>
      <td>both</td>
    </tr>
    <tr>
      <th>49107</th>
      <td>272699</td>
      <td>2009</td>
      <td>2009-06-30</td>
      <td>92155.0</td>
      <td>20</td>
      <td>3690.0</td>
      <td>369.0</td>
      <td>2.0</td>
      <td>ZBB</td>
      <td>WI</td>
      <td>...</td>
      <td>0.094216</td>
      <td>0.131838</td>
      <td>-0.057063</td>
      <td>0.094087</td>
      <td>0.034401</td>
      <td>1.718830</td>
      <td>4.873670</td>
      <td>1</td>
      <td>0.536936</td>
      <td>both</td>
    </tr>
    <tr>
      <th>49118</th>
      <td>284041</td>
      <td>2010</td>
      <td>2011-02-28</td>
      <td>93355.0</td>
      <td>45</td>
      <td>3572.0</td>
      <td>357.0</td>
      <td>0.0</td>
      <td>OCZTQ</td>
      <td>CA</td>
      <td>...</td>
      <td>0.008012</td>
      <td>0.048776</td>
      <td>-0.023342</td>
      <td>0.041962</td>
      <td>0.352064</td>
      <td>1.397729</td>
      <td>1.976579</td>
      <td>8</td>
      <td>0.876396</td>
      <td>both</td>
    </tr>
  </tbody>
</table>
<p>36057 rows × 46 columns</p>
</div>




```python
subsample.groupby(['gsector','fyear'])['RETech_avg'].mean()
```




    gsector  fyear
    15       1980     1.396836
             1981     1.819948
             1982     1.455234
             1983     1.647780
             1984     1.547525
                        ...   
    45       2006     1.200705
             2007     1.299859
             2008     0.916899
             2009     0.045175
             2010     1.033466
    Name: RETech_avg, Length: 186, dtype: float64



- Q9. Print out the year 2000 industry averages you just computed.


```python
subsample[subsample['fyear'] == 2000].groupby(['gsector'])['RETech_avg'].mean()
```




    gsector
    15    0.521415
    20    0.668028
    25    1.114153
    35    1.420250
    40    2.438407
    45    2.161341
    Name: RETech_avg, dtype: float64



- Q10. Plot the time-trends of the industry averages 
    - 2%: Set the title, xlabel, and ylabel
    - 2%: There should be no error bands showing
    - 4%: Replace the gsector numbers with the names of the industries
    - 3% of the total grade of the assignment will be reserved for implementing [the "sparkline" style of graph ](https://github.com/orgs/LeDataSciFi/teams/classmates-2023/discussions/11/comments/2) several students made for Assignment 3.
        - Because of how I chose to structure the data, I had to alter the code linked above to use hue instead of units, and [then correct the background line colors](https://stackoverflow.com/questions/67221399/plotting-multiple-lines-with-same-color-but-using-hue-to-separate-the-lines).
        - It's possible my tweaks can be avoided. 


```python
pt2 = subsample.groupby(['gsector','fyear'])['RETech_avg'].mean()

data = pd.DataFrame(pt2)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>RETech_avg</th>
    </tr>
    <tr>
      <th>gsector</th>
      <th>fyear</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">15</th>
      <th>1980</th>
      <td>1.396836</td>
    </tr>
    <tr>
      <th>1981</th>
      <td>1.819948</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>1.455234</td>
    </tr>
    <tr>
      <th>1983</th>
      <td>1.647780</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>1.547525</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">45</th>
      <th>2006</th>
      <td>1.200705</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>1.299859</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>0.916899</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>0.045175</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>1.033466</td>
    </tr>
  </tbody>
</table>
<p>186 rows × 1 columns</p>
</div>




```python
g = sns.relplot(
    data=data,
    x='fyear', y='RETech_avg', col="gsector", hue="gsector",
    kind="line", palette="crest", linewidth=3, zorder=5,
    col_wrap=3, height=2, aspect=2, legend=False,
)

# Iterate over each subplot to customize further
for year, ax in g.axes_dict.items():
    
    # Plot every year's time series in the background
    sns.lineplot(
        data=data, x='fyear', y='RETech_avg', units="gsector",
        estimator=None, color=".7", linewidth=1, ax=ax
    )

# Replace the gsector numbers with the names of the industries 
g.axes[0].set_title('Materials')
g.axes[1].set_title('Industrials')
g.axes[2].set_title('Consumer Discretionary')
g.axes[3].set_title('Health Care')
g.axes[4].set_title('Financials')
g.axes[5].set_title('Information Technology')

# Reduce the frequency of the x axis ticks
ax.set_xticks(ax.get_xticks()[1:8:2])

# Tweak the supporting aspects of the plot
g.set_axis_labels("Fiscal Year", "RETech")
g.tight_layout()
g.fig.suptitle('Time-trends of industry averages for RETech (1980 - 2010)')
g.fig.subplots_adjust(top = 0.8)
```


    
![png](output_32_0.png)
    


## Part 3: Outliers

Let's consider if patent-level RETech should be winsorized for any analysis, and if so, how to define the limits. 

- Download the patent-level RETech data.


```python
#pip install wget

import wget
import os
```


```python
if not os.path.exists('Pat_text_vars_NotWinsored.zip'):
    url2 = 'https://github.com/donbowen/Patent-Text-Variables/releases/download/data-to-2018/Pat_text_vars_NotWinsored.zip'
    wget.download(url2)

zf = ZipFile('Pat_text_vars_NotWinsored.zip') 

df = pd.read_csv(zf.open('Pat_text_vars_NotWinsored.csv'))
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pnum</th>
      <th>RETech</th>
      <th>ayear</th>
      <th>Breadth</th>
      <th>gyear</th>
      <th>nber</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>948419</td>
      <td>5.328984</td>
      <td>1910</td>
      <td>0.488365</td>
      <td>1910</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>948819</td>
      <td>2.177653</td>
      <td>1910</td>
      <td>0.606481</td>
      <td>1910</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>948933</td>
      <td>2.690911</td>
      <td>1910</td>
      <td>0.512051</td>
      <td>1910</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>949133</td>
      <td>1.602156</td>
      <td>1910</td>
      <td>0.242670</td>
      <td>1910</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>949323</td>
      <td>3.692074</td>
      <td>1910</td>
      <td>0.285375</td>
      <td>1910</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9889599</th>
      <td>11212918</td>
      <td>0.577510</td>
      <td>2018</td>
      <td>0.462349</td>
      <td>2021</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9889600</th>
      <td>11212923</td>
      <td>0.546158</td>
      <td>2018</td>
      <td>0.545413</td>
      <td>2021</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9889601</th>
      <td>11212924</td>
      <td>1.233358</td>
      <td>2018</td>
      <td>0.532061</td>
      <td>2021</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>9889602</th>
      <td>11212936</td>
      <td>0.738356</td>
      <td>2018</td>
      <td>0.623213</td>
      <td>2021</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9889603</th>
      <td>11212944</td>
      <td>1.183381</td>
      <td>2018</td>
      <td>0.339578</td>
      <td>2021</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
<p>9889604 rows × 6 columns</p>
</div>



- Q11: Make one time-trend plot, covering all years in the data, with three lines:
    - Use the application year for each patent, not the grant year
    - Median RETech that year (blue line)
    - 99th percentile RETech that year (black line)
    - 99th percentile of RETech, a number computed over the entire sample at once (red line)


```python
year_median = df.groupby('ayear')['RETech'].median()
year_median
```




    ayear
    1910    2.440419
    1911    2.409563
    1912    2.213586
    1913    2.207886
    1914    2.255470
              ...   
    2014    0.835670
    2015    0.672422
    2016    0.827533
    2017    0.871436
    2018    1.386850
    Name: RETech, Length: 109, dtype: float64




```python
year_99q = df.groupby('ayear')['RETech'].quantile(.99)
year_99q
```




    ayear
    1910     9.916662
    1911    10.613583
    1912    10.749526
    1913     8.688449
    1914     8.219281
              ...    
    2014     5.493885
    2015     5.395979
    2016     6.225089
    2017     4.698825
    2018     5.648576
    Name: RETech, Length: 109, dtype: float64




```python
retech_99q = df['RETech'].quantile(.99)
retech_99q
```




    8.539103096274168




```python
sns.lineplot(data = year_median, color='blue', label= 'Median RETech')
sns.lineplot(data = year_99q, color='black', label = '99th Percentile RETech')
plt.axhline(y=retech_99q, color='r', linestyle='--', label = 'Cumulative 99th Percentile RETech')

plt.legend()
plt.xlabel('Application Year')
plt.title('Time trend plot of Median and 99th Percentile RETech for Patents filed (1910 - 2018)')
plt.show()
```


    
![png](output_40_0.png)
    


- Q12:  Short answer (<5 sentences): Discuss the difference for the red line and the black line, and what they imply for your choice of winsorization limits. 

The black line, or the cumulative 99th percentile RETech for the entire period (1910-2018), acts as a benchmark for comparing how RETech in patent applications have performed. For the period from 1910 -1960, the 99th percentile RETech is higher than the cumulative 99th percentile figure calculated for the entire period (1910-2018). After the 1960s, the trend for the 99th percentile RETech seems to be going down, with the exception of the mid-1990s, which goes in parallel with the technological changes that appeared in the late 1980s and mid-1990s and thus receiving patent applications for. Overall, it looks like the 99th percentile RETech for a given year is sloping downwards when compared with the cumulative 99th Percentile of RETech for the entire period.
