"""
File: conversation_server.py
"""
import sys
import socket
from _thread import start_new_thread
from tools.conversation_strategy import load
from tools.conversation_strategy import predict
SERVER_IP = '127.0.0.1'
SERVER_PORT = 8601
print('starting conversation server ...')
print('binding socket ...')
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind((SERVER_IP, SERVER_PORT))
except socket.error as msg:
    print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
    exit()
s.listen(10)
print('bind socket success !')
print('loading model...')
model = load()
print('load model success !')
print('start conversation server success !')


def clientthread(conn, addr):
    """
    client thread
    """
    logstr = 'addr:' + addr[0] + '_' + str(addr[1])
    try:
        param = conn.recv(4096).decode()
        logstr += '\tparam:' + param
        if param is not None:
            response = predict(model, param.strip())
            logstr += '\tresponse:' + response
            conn.sendall(response.encode())
        conn.close()
        print(logstr + '\n')
    except Exception as e:
        print(logstr + '\n', e)


while True:
    conn, addr = s.accept()
    start_new_thread(clientthread, (conn, addr))
s.close()
