import websocket, json
import pandas as pd


def generate_data():

    #Connect to the websocket of binance

    closes, highs, lows = [], [], []

    def on_message(ws, message):

        #Extracting data

        json_message = json.loads(message)
        candle = json_message['k']
        is_candle_closed = candle['x']
        close = candle['c']
        high = candle['h']
        low = candle['l']

        if is_candle_closed:
            closes.append(close)
            highs.append(high)
            lows.append(low)


    def on_close(ws):
        print('Conection Closed')


    socket = 'wss://stream.binance.com:9443/ws/btcusdt@kline_1m'
    ws = websocket.WebSocketApp(socket,
                            on_message=on_message,
                            on_close=on_close)
    
    ws.run_forever()

    #make a csv file with the data

    data = pd.DataFrame({'Closes':closes,'Highs':highs, 'Lows':lows}).astype(float)
    data.to_csv('/home/cod3_breaker/portafolio/Prueba-trademate/data.csv',
                 index=False)
