from train import train
import torch.optim as optim
import numpy as np

F_optims={'Adam':optim.Adam,'Nadam':optim.NAdam,'SGD':optim.SGD}
LRS={'1e-3':1e-3,'1e-4':1e-4,'1e-5':1e-5}
dic={}
i=0
for optim in F_optims:
    for lr in LRS:
        dic[optim+lr+'train_acc'],dic[optim+lr+'train_loss'],dic[optim+lr+'test_acc']=\
            train(F_optims[optim],LRS[lr],gamma=1,batch_size=32,cycles=10,MAX_EPOCH=3)
        np.save('/home/zy/Hands-on-RL/Resnet/'+str(i)+'dic.npy',dic)
        i+=1
        print('-'*80)

np.save('/home/zy/Hands-on-RL/Resnet/dic.npy',dic)