import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import os
from tabulate import tabulate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
keras = tf.keras

SYMBOLS = ['^DJI', '^GSPC', 'BTC-USD', 'DOGE-USD', 'TSLA', 'AAPL', 'MSFT', 'GC=F', 'SI=F', 'ETH-USD', 'ADA-USD', 'PG']
WINDOW_SIZE = 30

# Load Model
model = keras.models.load_model("stable_models/cnn_checkpoint.h5")
output_table=[]
column=1
for SYMBOL in SYMBOLS:
    start = dt.datetime(1970,1,1)
    end = dt.datetime(2022,1,1)
    data = web.DataReader(SYMBOL, 'yahoo', start=start, end=end)
    test_start = dt.datetime(2022,1,1)
    test_end = dt.datetime.now()

    test_data=web.DataReader(SYMBOL, 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - WINDOW_SIZE:].values
    model_inputs = model_inputs.reshape(-1,1)

    real_data = [model_inputs[len(model_inputs)+1 - WINDOW_SIZE:len(model_inputs+1),0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

    prediction = model.predict(real_data)

    convert_symbol = {"^DJI":"DOW", "^GSPC":"S&P500", "GC=F":"GOLD", "SI=F":"SILVER"}

    output_table.append([f"{SYMBOL}" if SYMBOL not in convert_symbol else f"{convert_symbol[SYMBOL]}",f"${test_data['Close'][-1]:.4f}",f"${prediction[0][-1][0]:.4f}"])
print(tabulate(output_table, headers=["Symbol", "Current Price", "Predicted Price"], tablefmt="psql"))