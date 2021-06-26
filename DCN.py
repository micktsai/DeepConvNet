import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import data.dataloader as DL

class DCN(nn.Module):
    def __init__(self):
        super(DCN, self).__init__()

        # L-1
        self.l1_conv1 = nn.Conv2d(1, 25,kernel_size=(1,5), bias=False)
        self.l1_conv2 = nn.Conv2d(25, 25,kernel_size=(2,1), bias=False)
        self.l1_batchnorm = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.l1_pooling = nn.MaxPool2d(kernel_size=(1,2), padding=0)
        # L-2
        self.l2_conv = nn.Conv2d(25, 50,kernel_size=(1,5), groups=25, bias=False)
        self.l2_batchnorm = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.l2_pooling = nn.MaxPool2d(kernel_size=(1,2), padding=0)
        # L-3
        self.l3_conv = nn.Conv2d(50, 100,kernel_size=(1,5), groups=50, bias=False)
        self.l3_batchnorm = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.l3_pooling = nn.MaxPool2d(kernel_size=(1,2), padding=0)
        # L-4
        self.l4_conv = nn.Conv2d(100, 200,kernel_size=(1,5), groups=100, bias=False)
        self.l4_batchnorm = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.l4_pooling = nn.MaxPool2d(kernel_size=(1,2), padding=0)
        # FC
        self.fc1 = nn.Linear(in_features=200*43, out_features=2, bias=True)
    def forward(self, x):
        x = x.clone().detach()
        # x = torch.tensor(x)
        # L-1
        x = self.l1_conv1(x)
        x = self.l1_conv2(x)
        x = self.l1_batchnorm(x)
        x = F.elu(x)
        x = self.l1_pooling(x)
        x = F.dropout(x, p=0.5)
        # L-2
        x = self.l2_conv(x)
        x = self.l2_batchnorm(x)
        x = F.elu(x)
        x = self.l2_pooling(x)
        x = F.dropout(x, p=0.5)
        # L-3
        x = self.l3_conv(x)
        x = self.l3_batchnorm(x)
        x = F.elu(x)
        x = self.l3_pooling(x)
        x = F.dropout(x, p=0.5)
        # L-4
        x = self.l4_conv(x)
        x = self.l4_batchnorm(x)
        x = F.elu(x)
        x = self.l4_pooling(x)
        x = F.dropout(x, p=0.5)
        # FC
        x = x.view(-1,200*43)
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        x = torch.argmax(x, dim=1)
        x = x.view(-1,1)
        return x

def evaluate(model, X, Y, params = ["acc"]):
    results = []
    batch_size = 36
    
    predicted = []
    
    for i in range(len(X)//batch_size):
        s = i*batch_size
        e = i*batch_size+batch_size
        
        inputs = Variable(torch.from_numpy(X[s:e]))
        pred = model(inputs)
        
        predicted.append(pred.data.cpu().numpy())
        
        
    inputs = Variable(torch.from_numpy(X))
    predicted = model(inputs)
    
    predicted = predicted.data.cpu().numpy()
    
    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2*precision*recall/ (precision+recall))
    return results

def main():
    train_data, train_label, test_data, test_label = DL.read_bci_data()
    net = DCN().double()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())
    # print(net(train_data[0:10]))
    # net(train_data)

    batch_size = 36

    for epoch in range(10):  # loop over the dataset multiple times
        print("Epoch ", epoch)
        
        running_loss = 0.0
        for i in range(int(len(train_data)/batch_size)-1):
            s = i*batch_size
            e = i*batch_size+batch_size
            
            inputs = torch.from_numpy(train_data[s:e])
            labels = torch.FloatTensor(np.array([train_label[s:e]]).T*1.0)
            
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs)
            # print(labels)
            loss = criterion(outputs.float(), labels.float())
            loss = loss.requires_grad_()
            loss.reduction = 'sum'
            # print(loss)
            loss.backward()
            
            
            optimizer.step()
            running_loss += loss.item()
        
        # Validation accuracy
        params = ["acc", "auc", "fmeasure"]
        print (params)
        print ("Training Loss ", running_loss)
        print ("Train - ", evaluate(net, train_data, train_label, params))
        # print ("Validation - ", evaluate(net, X_val, y_val, params))
        print ("Test - ", evaluate(net, test_data, test_label, params))

if __name__ =='__main__':
    main()