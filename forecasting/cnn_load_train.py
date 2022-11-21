import numpy as np
import tensorflow as tf
import pandas as pd
import pandas_datareader as web
import datetime as dt
import os
import random as r
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
keras = tf.keras

# Constants
SYMBOLS = ['^DJI', 'BTC-USD', 'DOGE-USD', 'TSLA', 'AAPL', '^GSPC', 'GC=F', 'ETH-USD', 'ADA-USD', 'PG']
EPOCHS = 75
BATCH_SIZE = 32
WINDOW_SIZE = r.randint(20,120)

# Helper Methods
def seq2seq_window_dataset(series, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(WINDOW_SIZE + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(WINDOW_SIZE + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(BATCH_SIZE).prefetch(1)

# Load Model
model = keras.models.load_model("prototype_models/cnn_checkpoint.h5")

model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="retrain_models/cnn_checkpoint.h5", save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(patience=13)
r.shuffle(SYMBOLS)
for SYMBOL in SYMBOLS:
    # Get Data from Yahoo Finance API
    start = dt.datetime(1970,1,1)
    end = dt.datetime(2022,1,1)
    data = web.DataReader(SYMBOL, 'yahoo', start=start, end=end)
    frame = pd.DataFrame(data)
    frame.reset_index(inplace=True,drop=False)
    date = frame['Date'][0]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

    time = len(data)
    buffer_size = int(time)
    split_time = int(buffer_size*0.8)
    print(f"Window: {WINDOW_SIZE}")
    print(f"Days: {buffer_size}")
    print(f"Split: {split_time}")
    print(f"Current: {data['Close'].values[-1]}")
    time = np.arange(buffer_size)

    #reshaped_data = data['Close'].values.reshape(-1,1)
    series=[]
    for i in scaled_data:
        series.append(i[0])
    series=np.array(series)

    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    train_set = seq2seq_window_dataset(x_train, buffer_size)
    valid_set = seq2seq_window_dataset(x_valid, buffer_size-split_time)

    model.fit(train_set, epochs=EPOCHS, validation_data=valid_set, callbacks=[early_stopping, model_checkpoint])

    # Display Forecast
    test_start = dt.datetime(2022,1,1)
    test_end = dt.datetime.now()

    test_data=web.DataReader(SYMBOL, 'yahoo', test_start, test_end)

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - WINDOW_SIZE:].values
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.transform(model_inputs)

    real_data = [model_inputs[len(model_inputs)+1 - WINDOW_SIZE:len(model_inputs+1),0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction[-1])
    print(f"{SYMBOL} Prediction: {prediction[-1]}")