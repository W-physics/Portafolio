import json
import websocket

from model import splitting
from model import preprocessing
from model import training_model
from model import make_order

def main():

    def on_message(ws, message):

        json_message = json.loads(message)
        candle = json_message['k']
#        is_candle_closed = candle['x']
        close = float(candle['c'])
        high = float(candle['h'])
        low = float(candle['l'])

        #Make a limit order based on ML predictions

#        if is_candle_closed:

        execute_order(close, high, low)

    def on_close(ws):
        print('Conection Closed')

    #Connecting to websocket

    socket = 'wss://stream.binance.com:9443/ws/btcusdt@kline_1m'
    ws = websocket.WebSocketApp(socket,
                                on_message=on_message,
                                on_close=on_close)
        
    ws.run_forever()

def execute_order(close, high, low):
        
        X_train, X_test, y_train, y_test = splitting()

        X_predict = preprocessing([close, high, low])

        model = training_model(X_train, y_train)

        order = model.predict(X_predict)

        if order == 1: 
            price = close - 50
            make_order('SELL', str(price))
            print(f'Sold at : {price}')
        else:
            price = close + 50  
            make_order('BUY', str(price))
            print(f'Bought at : {price}')

main()