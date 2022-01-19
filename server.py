# Depending on the no. of servers (n), run server.py with argument 0,1,...n, simultaneously in different terminals.


import socket
import pickle
import threading
import numpy as np
import sys
from fixedpoint import *
import math

def f_to_i(x, scale=1 << 32):
  if x < 0:
    if pow(2, 64) - (abs(x)*(scale)) > (pow(2, 64) - 1):
      return np.uint64(0)
    x = pow(2, 64) - np.uint64(abs(x)*(scale))

  else:
    x = np.uint64(scale*x)

  return np.uint64(x)


def i_to_f(x, scale=1 << 32):
  l = 64
  t = x > (pow(2, (l-1)) - 1)
  if t:
    x = pow(2, l) - x
    y = np.uint64(x)
    y = np.float32(y*(-1))/scale

  else:
    y = np.float32(np.uint64(x))/scale

  return y


f_to_i_v = np.vectorize(f_to_i)
i_to_f_v = np.vectorize(i_to_f)

class Config:
	weights = {}  # defining a dummy weight matrix
	count = 0
	S = {}
	Final = np.array([])
	check = False
	num_clients = 3
	num_servers = 3
	stop_receive_blocks = False
	count_sum = 0
test_class=Config()
no_of_layers=10 # change depending on model

serverNum = int(sys.argv[1]) # the command to run server.py needs to be (sudo) python3 server.py 0/1/2
PORT = 8500+serverNum
print("Assigning port: ",PORT)
IP = socket.gethostbyname(socket.gethostname())
print(IP)
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
serversocket.bind((IP, PORT)) # binds to the server socket to receive from clients 
SERVER = socket.gethostbyname(socket.gethostname())

def handle_client(conn, addr): # to handle multiple clients

	# global weights
	clientsocket, address = conn,addr
	ip = str(address[0])
	reply = "\n\nThank you for connecting to SERVER.\n Your ip address is: " + ip + \
			"\n Please send affirmation that you want to proceed with 1/0 \n " # sends reponse on getting connection request

	
	clientsocket.send(reply.encode()) # sends 1st response to the reqesting client

	flag = clientsocket.recv(1024).decode() # receiving reply

	if flag == '1':# receives message/data from the connected client
		received_data = b''
		while str(received_data)[-2] != '.':
			data = clientsocket.recv(4096)
			received_data += data
		print("Received all packets")
  
		data_arr = pickle.loads(received_data)
		no_of_layers=len(data_arr)
		if test_class.count==0:
			for i in range(no_of_layers):
				shape=data_arr[i].shape
				test_class.weights[i]=np.random.random_sample((Config.num_clients,)+shape)
				test_class.weights[i][test_class.count]=data_arr[i] # adding the share x_i to the existing weights array
		else:
			for i in range(no_of_layers):
				test_class.weights[i][test_class.count]=data_arr[i]
		test_class.count += 1
		
		if(test_class.count == test_class.num_clients):
			for i in range(no_of_layers):
				test_class.weights[i] = np.array((f_to_i_v(
                                    test_class.weights[i]*(i_to_f_v(float(1)/test_class.num_clients)))), dtype=np.uint64)
				test_class.S[i]=test_class.weights[i][0]
				for j in range(1,test_class.weights[i].shape[0]):
					test_class.S[i]+=test_class.weights[i][j]

			test_class.Final = test_class.S
			test_class.check=True
		# Similarly, can now send test_class.Final back to clients

	else:
		print("invalid choice")



def start():

	serversocket.listen()

	print(f"[LISTENING] Server is listening on {SERVER}")

	while True:
		conn, addr = serversocket.accept()
		thread = threading.Thread(target=handle_client, args=(conn, addr))
		thread.start()
		print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


print("[STARTING] server is starting...")
start()
