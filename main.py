import pandas as pd
import yfinance as yf
import numpy as np
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt

symbols = ["^NSEBANK","ICICIBANK.NS","HDFCBANK.NS","SBIN.NS"]

end_date = datetime.datetime.now().date()
start_date = end_date - datetime.timedelta(days=59)

dataBnf= yf.download("^NSEBANK", start=start_date, end=end_date, interval='5m')
dataHdfc= yf.download("HDFCBANK.NS", start=start_date, end=end_date, interval='5m')
dataIcici= yf.download("ICICIBANK.NS", start=start_date, end=end_date, interval='5m')
dataSbin= yf.download("SBIN.NS", start=start_date, end=end_date, interval='5m')

dataBnfReturns=(dataBnf["Close"]-dataBnf["Open"])/dataBnf["Open"]
dataHdfcReturns=(dataHdfc["Close"]-dataHdfc["Open"])/dataHdfc["Open"]
dataIciciReturns=(dataIcici["Close"]-dataIcici["Open"])/dataIcici["Open"]
dataSbinReturns=(dataSbin["Close"]-dataSbin["Open"])/dataSbin["Open"]
rolling_window=20
rolling_results=[]
y_R=pd.DataFrame({'dataBnfReturns':dataBnfReturns})
x=pd.DataFrame({'dataHdfcReturn':dataHdfcReturns,'dataIciciReturns': dataIciciReturns,'dataSbinReturns': dataSbinReturns})

for i in range(rolling_window, len(dataSbin)):
    y = dataBnfReturns.iloc[i - rolling_window:i]
    X = x.iloc[i - rolling_window:i]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    rolling_results.append(model.params)

rolling_results_df = pd.DataFrame(rolling_results, columns=['const', 'dataHdfcReturn', 'dataIciciReturns', 'dataSbinReturns'])

x["const"] = 1

y=dataBnfReturns
z = pd.DataFrame(columns=['Datetime', 'Z-score'])
for i in range(len(x) - 21):
    temp_x = x[i:i + 20]
    y_hat = temp_x.mul(rolling_results_df.iloc[i], 1).sum(axis=1)
    temp_y = y[i:i + 20]

    error = y_hat - temp_y
    e_bar = error.mean()
    e_std = error.std()

    x_21 = x.iloc[i + 21]
    y_21 = y.iloc[i + 21]
    y_pred = x_21.mul(rolling_results_df.iloc[i], 1).sum()
    error_21 = y_pred - y_21
    z = z.dropna(how='all')
    z = z._append({'Datetime': y.index[i + 21], 'Z-score': (error_21 - e_bar) / e_std}, ignore_index=True)

fin=pd.merge(z,x[21:len(x)],on='Datetime')
print(y_R[21:len(y_R)])
trie=pd.merge(fin,y_R[21:len(x)],on='Datetime')
trie.to_csv('zscores.csv')
print(trie)

position="None"
sum=0
result=0
tradebook=[]
for index, row in trie.iterrows():
    if position=='None':
        sum=0
        if row['Z-score'] >= 2.0:
            entry_time = row['Datetime']
            position = 'Short'

        if row['Z-score'] <= -2.0:
            entry_time = row['Datetime']
            position = 'Long'
        continue

    elif position=='Short':
        if row['Z-score'] <= 0:
            exit_time = row['Datetime']
            result+=sum
            tradebook.append({'Entry Time': entry_time, 'Exit Time': exit_time, 'Position': 'short', 'return': sum})
            position = 'None'

        else:
            sum = -row['dataBnfReturns']
            sum += ((row['dataHdfcReturn'] + row['dataIciciReturns'] + row['dataSbinReturns'])/3)

    elif position=='Long':
        if row['Z-score'] >= 0:
            exit_time = row['Datetime']
            result += sum
            tradebook.append({'Entry Time': entry_time, 'Exit Time': exit_time, 'Position': 'Long', 'return': sum})
            position = 'None'

        else:
            sum = +row['dataBnfReturns']
            sum -= (row['dataHdfcReturn'] + row['dataIciciReturns'] + row['dataSbinReturns'])/3

print(tradebook)
tradebook_df = pd.DataFrame(tradebook)
print('final return',result)

tradebook_df['Entry Time'] = pd.to_datetime(tradebook_df['Entry Time'])
tradebook_df['Exit Time'] = pd.to_datetime(tradebook_df['Exit Time'])

tradebook_df['Day'] = tradebook_df['Entry Time'].dt.date

daily_returns = tradebook_df.groupby('Day')['return'].sum().reset_index()

daily_returns.rename(columns={'return': 'Daily Returns'}, inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(tradebook_df['Exit Time'], tradebook_df['capital'])
plt.xlabel('Time')
plt.ylabel('Capital')
plt.title('Capital Curve')
plt.grid(True)
plt.show()

risk_free_rate = 0.03 / 252

def calculate_metrics(tradebook_df):
    excess_return = tradebook_df['Daily Returns'] - risk_free_rate
    sharpe_ratio = np.sqrt(252) * excess_return.mean() / excess_return.std()

    tradebook_df['Cumulative Returns'] = (1 + tradebook_df['Daily Returns']).cumprod()
    tradebook_df['RollingMax'] = tradebook_df['Cumulative Returns'].cummax()
    tradebook_df['Drawdown'] = (tradebook_df['Cumulative Returns'] / tradebook_df['RollingMax']) - 1
    max_drawdown = tradebook_df['Drawdown'].min()

    return sharpe_ratio, max_drawdown

sharpe_ratio, max_drawdown = calculate_metrics(daily_returns)

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")

def calculate_sharpe_ratio(tradebook_df, lookback_window):
    tradebook_df['RollingAvgReturns'] = tradebook_df['Daily Returns'].rolling(lookback_window).mean()
    tradebook_df['RollingStdReturns'] = tradebook_df['Daily Returns'].rolling(lookback_window).std()
    excess_return = tradebook_df['RollingAvgReturns'] - risk_free_rate
    sharpe_ratio = np.sqrt(252) * excess_return / tradebook_df['RollingStdReturns']
    return sharpe_ratio

lookback_windows = [5, 10, 20]

sharpe_ratios = {}
for window in lookback_windows:
    sharpe_ratios[window] = calculate_sharpe_ratio(daily_returns, window)

plt.figure(figsize=(10, 6))
for window in lookback_windows:
    plt.plot(tradebook_df['Day'], sharpe_ratios[window], label=f'Lookback {window} days')

plt.xlabel('Date')
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratio vs. Lookback Window')
plt.legend()
plt.grid(True)
plt.show()
