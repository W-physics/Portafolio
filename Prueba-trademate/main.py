from binance.spot import Spot
import json
import websocket

from model import training_model
from model import splitting
from model import preprocessing

def main():

    X_train, X_test, y_train, y_test = splitting()

    model = training_model(X_train, y_train)

    def on_message(ws, message):

        json_message = json.loads(message)
        candle = json_message['k']
        is_candle_closed = candle['x']
        close = candle['c']
        high = candle['h']
        low = candle['l']

        #Make a limit order based on ML predictions

        if is_candle_closed:

            #Conver to float

            close = close.astype(float)
            high = high.astype(float)
            low = low.astype(float)

            X_predict = preprocessing([close, high, low])

            order = model.predict(X_predict)

            if order == 0: 
                price = close - 50
                make_order('SELL', str(price))
                print(f'Sold at : {price}')
            else:
                price = close + 50
                make_order('BUY', str(price))
                print(f'Bought at : {price}')
                
    def on_close(ws):
        print('Conection Closed')

    socket = 'wss://stream.binance.com:9443/ws/btcusdt@kline_1m'
    ws = websocket.WebSocketApp(socket,
                                on_message=on_message,
                                on_close=on_close)
        
    ws.run_forever()

def logging():

    client = Spot(api_key='9bWvHb3DaWCq9JSp36bieIAb0yxIvf68aBb3xS5ZHOEu2GgwPKHAj7uNpErrDPgk'
              ,api_secret='cWOP89mPckrKO6ykdiEQENdQjU5aMtZnZZVeinwlXtvAsxrRb8ZSU3Xaf76ZJuk1')

    return client

#Make a limit order

def make_order(order, price):

    client = logging()

    order_limit = {'symbol':'BTCUSDT',
               'side': order,
               'type': 'LIMIT',
               'price': price,
               'quantity': '0.0001',
               'timeInForce': 'GTC'}

    client.new_order_test(**order_limit)

main()