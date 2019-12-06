import pandas as pd
import requests 
import datetime
import matplotlib.pyplot  as plt

#Test Alpha Vantage 
#APIKEY: 79DMF9EARFU9LM89

class Stock_Data:
    def api_key(self,api_key,loc):
        self.api_key = api_key
        api_key=open('alpha.txt','r').read()

#api_key= 79DMF9EARFU9LM89
        
    data=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&&outputsize=full&symbol=GE&apikey={}'.format(api_key))
    data=data.json()
    data=data['Time Series (Daily)']

    #def stock_df(self,data,loc):
    stock_df = pd.DataFrame(columns=["date","open","high","low","close","volume"])
    stock_df=stock_df.iloc[1:]
    for d,p in data.items():
        date = datetime.datetime.strptime(d,'%Y-%m-%d')
        data_row=[date,float(p['1. open']),float(p['2. high']),float(p['3. low']),float(p['4. close']),int(p['5. volume'])]
        stock_df.loc[-1,:] = data_row
        stock_df.index = stock_df.index+1
    data=stock_df.sort_values(['date'], axis=0,ascending = True, inplace=True)
    
    #Add 5 day moving average
    Ma5 = stock_df.close.rolling(window=5).mean()
    stock_df['Ma5'] = Ma5

        
    #Add turover ration
        
    Volume_Turover_Ratio = stock_df.volume.pct_change()/100
    Volume_Turover_Ratio = Volume_Turover_Ratio.abs()
    stock_df['Volume_Turover_Ratio'] = Volume_Turover_Ratio
    print(stock_df)
    #export_csv = stock_df.to_csv (r'C:\Users\Saran\Desktop\export_dataframe.csv', index = None, header=True) 
    
    plt.figure(figsize = (16,9))
    plt.plot(range(stock_df.shape[0]),(stock_df['low']+stock_df['high'])/2.0)
    plt.xticks(range(0,stock_df.shape[0],500),stock_df['date'].loc[::500],rotation=45)
    plt.xlabel('date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.show()



