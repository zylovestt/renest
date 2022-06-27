'''
This program is part of the teaching materials for teacher Hao Xiaoli's experimental class of BJTU.

Copyright © 2021 HAO xiaoli and CHANG tianxing.
All rights reserved.
'''
from cProfile import label
import os
from pickletools import optimize
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from resnet18 import ResNet18_model as ResNet_model
import DogCatDataset

def cal_test_acc(net,test_data,device):
    num=0
    right=0
    for data in test_data:
        pre=net(data[0].to(device)).argmax(axis=1)
        right+=(pre==data[1].to(device)).sum().item()
        num+=data[0].shape[0]
    return right/num

def train(F_optim=optim.Adam,lr=0.001,gamma=0.9,batch_size=2,cycles=10,MAX_EPOCH = 100):
    # Step 0:查看torch版本、设置device
    print('Pytorch Version = ', torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1:准备数据集
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_data = DogCatDataset.DogCatDataset(root_path='/workspace/data/train_set',
                                             transform=train_transform)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = DogCatDataset.DogCatDataset(root_path='/workspace/data/test_set',
                                             transform=train_transform)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # Step 2: 初始化模型，请完善代码
    model=ResNet_model(num_classes=2)
    model.to(device)

    # Step 3:交叉熵损失函数，请完善代码
    criterion=nn.CrossEntropyLoss()

    # Step 4:选择优化器，请完善代码
    #optimizer=optim.SGD(model.parameters(),lr=LR,momentum=0.9)
    optimizer=F_optim(model.parameters(),lr=lr)
    # Step 5:设置学习率下降策略，请完善代码
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=gamma)

    # Step 6:训练网络，模型可以保存在 /workspace/weights 
    if not os.path.exists('/workspace/weights'):
        os.makedirs('/workspace/weights')
    
    model.train()
    train_loss=[]
    train_acc=[]
    test_acc=[]
    for epoch in range(MAX_EPOCH):
        loss_total = 0
        total_sample = 0
        accuracy_total = 0
        for iteration, data in enumerate(train_dataloader):
            #print('%%{}'.format(iteration*100/len(train_data)))
            # 请完善训练代码
            img,label=data
            img,label=img.to(device),label.to(device)
            output=model(img)
            optimizer.zero_grad()
            loss=criterion(output,label)
            loss.backward()
            optimizer.step()
            _,predicted_label=torch.max(output,1)
            total_sample+=label.size(0)
            accuracy_total+=torch.mean((predicted_label==label).type(torch.FloatTensor)).item()
            loss_total+=loss.item()

            if (iteration+1)%cycles==0:
                train_loss.append(loss_total/cycles)
                acc=accuracy_total/cycles
                train_acc.append(acc)
                test_acc.append(cal_test_acc(model,test_dataloader,device))
                loss_total = 0
                total_sample = 0
                accuracy_total = 0
                print('epoch:{},iteration:{},max_iter:{},train_loss:{},train_acc:{},test_acc:{},'.format(
                    epoch+1,iteration+1,len(train_data)/batch_size,train_loss[-1],train_acc[-1],test_acc[-1]))
                
        # 更新学习率，请完善代码
        scheduler.step()  

    return train_acc,train_loss,test_acc
