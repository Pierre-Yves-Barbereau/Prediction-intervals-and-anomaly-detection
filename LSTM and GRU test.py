# Databricks notebook source
# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/preprocessing

# COMMAND ----------

path_notebook_preproc_preprocessing = "/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing"
preproc = preprocessing(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
df_train_loaded = preproc.load_train("DecPDV")

# COMMAND ----------

df_train = df_train_loaded
index_df_train = df_train["DT_VALR"]
df_train = df_train.drop(["DT_VALR"],axis = 1)

# COMMAND ----------

"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
for column in df_train.columns:
  scaler.fit(df_train[column].values.reshape(-1,1))
  df_train[column] = scaler.transform(df_train[column].values.reshape(-1,1))
  """

# COMMAND ----------

target = df_train["Valeur"]
df_train = df_train.drop(["Valeur"],axis = 1)

# COMMAND ----------

display(df_train)

# COMMAND ----------

X_train,X_test,Y_train,Y_test = train_test_split(df_train, target, test_size=0.33, random_state=42)

# COMMAND ----------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR

# COMMAND ----------

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # we need to detach h0 and c0 here since we are doing back propagation through time (BPTT)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out
      

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # we need to detach h0 here since we are doing back propagation through time (BPTT)
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out
      
def train(net, x_train, y_train, x_test, y_test, criterion, optimizer):
  print(net)
  net.to(device)
  start_time = time.time()

  hist = []

  for t in range(num_epochs):
    
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    y_train_pred = net(x_train)
    loss = criterion(y_train_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    hist.append(loss.item())
    if t !=0 and t % 100 == 0 :
      print(' Epoch: {:.6f} \t Training loss: {:.6f} ' .format(t, loss.item()))
      test_loss = criterion(net(x_test.to(device)), y_test.to(device)).item()
      print(' Epoch: {:.6f} \t Test loss: {:.6f} ' .format(t, test_loss))
      if t % 1000 == 0:
        scheduler.step()
  training_time = time.time()-start_time
  print("Training time: {}".format(training_time))

  return np.array(hist)

# COMMAND ----------

input_dim = 1     # Number of features/columns used in training/testing. If we plann to use close price and open price for prediction, this would be changed to 2
hidden_dim = 32   # Hidden dimension
num_layers = 1    # Number for LSTM/GRU layer(s) used. We are using only 1. If we plan to use bi-directional RNN then this would be changed to 2
output_dim = 1    # Dimension of the output we are trying to predict (either close price/ open price/ high / low)
num_epochs = 10000 # we train our LSTM/GRU models for 10000 epochs
# we use Adam Optimizer and MSE loss fucntion for trainig the models

# COMMAND ----------

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# COMMAND ----------

Y_train

# COMMAND ----------



# COMMAND ----------

X_train_tensor = torch.from_numpy(np.array(X_train).astype(float)).type(torch.Tensor).unsqueeze(2)
X_test_tensor = torch.from_numpy(np.array(X_test).astype(float)).type(torch.Tensor).unsqueeze(2)
Y_train_tensor = torch.from_numpy(np.array(Y_train).astype(float)).type(torch.Tensor).unsqueeze(1)
Y_test_tensor = torch.from_numpy(np.array(Y_test).astype(float)).type(torch.Tensor).unsqueeze(1)

# COMMAND ----------

net_lstm_h = LSTM(input_dim, hidden_dim, num_layers, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(net_lstm_h.parameters(), lr=100000)
scheduler = ExponentialLR(optimizer, gamma=0.9)
hist_lstm = train(net_lstm_h, X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, criterion, optimizer)

# COMMAND ----------

