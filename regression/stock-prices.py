import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style


style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

subset_features = ['Adj. Open', 'Adj. High','Adj. Low','Adj. Close', 'Adj. Volume']
df = df[subset_features]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# label for this regression is the 'close' price some days (1%)
# into the future; can't use the 'close' price of the same day!
label = 'Future Price'
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
print("Taking labels as closing price for " + str(forecast_out) + " days in the future.")
df[label] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop([label], 1)) # returns new df
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
# no need to take care of the shifting since we dropped na
y = np.array(df[label])

#print(len(X), len(y))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

classifier = LinearRegression(n_jobs=-1) # run as many threads as possible
#classifier = svm.SVR(kernel='poly')
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
# print(accuracy)

forecast_set = classifier.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
print(last_date, type(last_date))
#last_unix = last_date.timestamp()
#last_unix = last_date.view('int64')
last_unix = last_date.value / 1000000000 # in seconds
print(last_unix)
one_day = 86400
next_unix = last_unix + one_day
print(datetime.datetime.fromtimestamp(next_unix))

# this is to fill in date values for the last 'forecast_out' rows
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	#next_date_pd = pd.to_datetime(next_date, unit='s')
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

