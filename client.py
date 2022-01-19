# Run the client.py file for each client with just one change: weights file name.
# For eg., if there are 3 clients, run this file 3 times with the corresponding weight file names.
# follow the prompts appearing each time.

import socket
import pickle 
import numpy as np
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
# read dataset
infile = open("test_weights.pkl",'rb')
data = pickle.load(infile)
infile.close()
# read dataset
layer_dict,layer_shape,shares_dict={},{},{}

no_of_layers=len(data)
for i in range(len(data)):
    layer_dict[i]=data[i]
    layer_shape[i]=data[i].shape
    
for i in range(no_of_layers):
    shares_dict[i]=np.random.random_sample((Config.num_servers,)+layer_shape[i])
    x=layer_dict[i]
    for k in range(0,Config.num_servers-2):
        shares_dict[i][k]=np.random.random_sample(layer_shape[i])
        x = x - shares_dict[i][k]
    shares_dict[i][Config.num_servers-1] = x
    print(shares_dict[i].shape)
for i in range(Config.num_servers):# iterating over the server ports 
    clientsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM) #connect via TCP
    # connect to the ith server  in the list of servers
    clientsocket.connect(('192.168.0.1', (8500+i)))


    reply = clientsocket.recv(1024).decode() #receiving server response
    print(reply) # printing the reply received from server

    choice_made = str(input("Please enter the appropriate choice accordingly: ")) # making the appropriate choice
    clientsocket.send(choice_made.encode()) # sending the answer


    print(f"the choice_made was {choice_made}") # option chosen by the user


    if int(choice_made)==1: 
        if i==Config.num_servers-1:
            test_dict={}
            for k in range(no_of_layers):
                test_dict[k]=f_to_i_v(shares_dict[k][i])
            msg = pickle.dumps(test_dict) 
        else:
            test_dict={}
            for k in range(no_of_layers):
                test_dict[k] = f_to_i_v(shares_dict[k][i])
            msg = pickle.dumps(test_dict)  
        clientsocket.sendall(msg)
        print(f"sent x{i} to server")


    elif int(choice_made)==2:
        received_data = b''
        while str(received_data)[-2] != '.':
            data = clientsocket.recv(4096)
            received_data += data
        print("Received all packets")
        updated_weights = pickle.loads(received_data)

    else:
        print("didnt get appropriate choice ")
        clientsocket.close()
