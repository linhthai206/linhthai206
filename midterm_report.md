# Summary

In this Assignment 5, I downloaded the stocks information for S&P500 companies. I downloaded their 10-K filing documents, stock returns, calculated the sentiment scores based on 10 sentiment variables (4 pre-existing sources and 6 self-generated ones on the topics of ESG, COVID, and climate change). The analysis discusses whether there is any correlation or relationship between the firm's stock returns and its sentiment analysis surrounding a chosen topic.

# Data

The sample of firms is taken from the S&P500 firm list on Wikipedia. 

- The return variables are calculated by sectioning the original stock returns dataframe to include only 10 days from the filing date. I further sectioned this new dataframe into two new ones, one including the filing date and 2 days from it, and the other is the remaining days with the stock returns. I then calculated the cumulative returns for each version using the method used in Assignment 2.

- The sentiment variables:
    - The first 4 variables (LM positive-negative and BHR positive-negative) are calculated by initially taking the lists from the original sources using ``pd.read_csv``. Then I used what I learned in class using REGEX to turn the respective lists into the format that regex wants with ``['('+'|'.join(happy_sentiment)+')']``). Then I proceeded to calculate the actual sentiment score with ``df1['LM_negative'] = df1['cleaned_html'].apply(lambda x:len(re.findall(NEAR_regex(LM_negative_regex),x)) /len(x.split()))``, essentially taking the entire LM_negative list and search for the chance of it appearing in the 10-K document. For the remaining 3 chosen topics, I generated my own list of negative and positive words to be used for each of my chosen topics (COVID, ESG, and climate change). Then I followed the same steps to convert these word lists into the same format and apply the same formula to calculate how much these topics get mentioned in the 10-K documents.
                                               
- Sentiment measure topics: I chose the following topics because I personally care about them. For COVID, I wanted to see how COVID's impact is still influencing companies and their financials in 2022 and whether it is still regarded as having negative influence. For climate change, I think that it is one of the currently most boiling issues of our time, and I want to see whether it is mentioned as having an impact on companies financials in their 10 case, and whether it is mentioned as a positive or negative phenomenon affecting the firms business. For E.S.G., I agree that E.S.G. includes the topic of climate change, however, it is broader and relates to the firms corporate social responsibility. Here I want to see if the analysis from E.S.G. may correlate with climate change, because they are relevant to each other, even though they don’t necessarily measure the same thing.



# Results

- One can make a couple observations based on the describe() table in the df1.describe() in the buid_sample_example file. BHR_positive has generally a lower mean than LM_positive, whereas BHR_negative has a higher mean than LM_negative. The documents on average have more words from the BHR list compared to LM list. For the chosen topics, firms seem to not explicit mention or talk about cimate change and its impact but more so ESG as a topic; they tend to mention ESG as a positive topic as opposed to climate change. THese topics get mentioned much less than COVID, which is mentioned as negative. BHR_negative has the highest standard deviation of all sentiment variables.

- Future analysis and improvements may include using a better and larger word list for the three chosen topics for sentiment analysis. 

# Code


```python
#if this doesn't run as is, please refer to build_sample_exercise file for the analysis and results
```


```python
import fnmatch
import glob
import os
import re
from time import sleep
from zipfile import ZipFile

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from near_regex import NEAR_regex  # copy this file into the asgn folder
from tqdm import tqdm  # progress bar on loops

# if you have tqdm issues, run this in terminal or with ! trick
# jupyter nbextension enable --py widgetsnbextension
# jupyter labextension install @jupyter-widgets/jupyterlab-manager
#
# if that fails, you can disable it

os.makedirs("output", exist_ok=True)

from sec_edgar_downloader import Downloader
import shutil

import warnings
warnings.filterwarnings("ignore", message="It looks like you're parsing an XML document using an HTML parser")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```


```python
sp500 = 'inputs/sp500.csv'

if not os.path.exists(sp500):
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    pd.read_html(url)[0].to_csv(sp500, index = False)
    
sp500 = pd.read_csv(sp500)
sp500
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
      <th>Symbol</th>
      <th>Security</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date added</th>
      <th>CIK</th>
      <th>Founded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1957-03-04</td>
      <td>66740</td>
      <td>1902</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1957-03-04</td>
      <td>1800</td>
      <td>1888</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
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
    </tr>
    <tr>
      <th>498</th>
      <td>YUM</td>
      <td>Yum! Brands</td>
      <td>Consumer Discretionary</td>
      <td>Restaurants</td>
      <td>Louisville, Kentucky</td>
      <td>1997-10-06</td>
      <td>1041061</td>
      <td>1997</td>
    </tr>
    <tr>
      <th>499</th>
      <td>ZBRA</td>
      <td>Zebra Technologies</td>
      <td>Information Technology</td>
      <td>Electronic Equipment &amp; Instruments</td>
      <td>Lincolnshire, Illinois</td>
      <td>2019-12-23</td>
      <td>877212</td>
      <td>1969</td>
    </tr>
    <tr>
      <th>500</th>
      <td>ZBH</td>
      <td>Zimmer Biomet</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>Warsaw, Indiana</td>
      <td>2001-08-07</td>
      <td>1136869</td>
      <td>1927</td>
    </tr>
    <tr>
      <th>501</th>
      <td>ZION</td>
      <td>Zions Bancorporation</td>
      <td>Financials</td>
      <td>Regional Banks</td>
      <td>Salt Lake City, Utah</td>
      <td>2001-06-22</td>
      <td>109380</td>
      <td>1873</td>
    </tr>
    <tr>
      <th>502</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
    </tr>
  </tbody>
</table>
<p>503 rows × 8 columns</p>
</div>




```python
dl = Downloader("10k_files")
```


```python
# assumption: if we have a zip file, it means we are done with downloads
# so don't download anything

if not os.path.exists('10k_files/10k_files.zip'):

    for firm in tqdm(sp500['Symbol']):

        firm_folder = "10k_files.zip/sec-edgar-filings/" + firm

        # if I haven't downloaded an HTML for this firm, do so
        if len(glob.glob(firm_folder + '/10-K/*/*.html')) == 0:
            dl.get("10-K", firm, amount=1, after="2022-01-01", before="2022-12-31")

        # pause - be nice to server 
        # NVM: not needed! sec_edgar_downloader automatically limits speed 

        # we don't need the .txt files. If there is one for this firm, delete it
        for txt_f in glob.glob(firm_folder + '/10-K/*/*.txt'):
            os.remove(txt_f)
```


```python
import zipfile
import os
from requests_html import HTMLSession
session = HTMLSession()

accession_list =[]
cik_list =[]
ticker_list=[]

zip_file = "10k_files/10k_files.zip"
folder_name = "sec-edgar-filings"

with zipfile.ZipFile(zip_file) as archive:
    # Loop through all files in the archive
    for file in archive.namelist():
        # Check if the file is a 10-K filing
        if "10-K" in file and file.endswith(".html"):
            # Extract CIK and accession number from the file path
            accession = os.path.basename(os.path.dirname(file))
            cik = accession.split("-")[0].lstrip("0")
            accession_list.append(accession)
            cik_list.append(cik)
            parts = file.split("/")
            ticker = parts[1]
            ticker_list.append(ticker)
```


```python
df1 = pd.DataFrame({'Symbol': ticker_list, 'Accession': accession_list})
df1
```


```python
df2 = sp500.merge(df1, on='Symbol')
df2
```


```python
df1= pd.DataFrame()
df1['CIK'] = np.array(cik_list)
df1['Accession Number'] =   np.array(accession_list)
df1
```


```python
cik_list1 =[]
filing_date_list=[]

cik_accession = list(zip(cik_list,accession_list))

for cik, accession_number in cik_accession:
    url = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}-index.html'
    r = session.get(url)
    filing_date = r.html.find('div.formContent > div:nth-child(1) > div:nth-child(2)', first=True).text.strip()
    cik_list1.append(cik)
    filing_date_list.append(filing_date)

df3= pd.DataFrame()
df3['CIK'] = np.array(cik_list1)
df3['Filing Date'] =   np.array(filing_date_list)
df3
```


```python
firms_df = pd.DataFrame(firms_df)
firms_df['ticker'] = np.array(ticker_list)
firms_df['filing date'] = np.array(ticker_list)
```


```python
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
```


```python
url = 'https://github.com/LeDataSciFi/data/raw/main/Stock%20Returns%20(CRSP)/crsp_2022_only.zip'

with urlopen(url) as request:
    data = BytesIO(request.read())

with ZipFile(data) as archive:
    with archive.open(archive.namelist()[0]) as stata:
        crsp1 = pd.read_stata(stata)
        
crsp1
```
