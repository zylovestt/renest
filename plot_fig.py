import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

dic=np.load('dic.npy',allow_pickle=True).item()
F_optims={'Adam':optim.Adam,'Nadam':optim.NAdam,'SGD':optim.SGD}
LRS={'1e-3':1e-3,'1e-4':1e-4,'1e-5':1e-5}
fig_names=['train_acc','train_loss','test_acc']

for name in fig_names:
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['figure.dpi']=300
    plt.rcParams['figure.figsize'] = (20,30)
    plt.rcParams.update({'font.size': 15})
    ynum,xnum=len(LRS),1
    f,axarr=plt.subplots(ynum,xnum)
    f.subplots_adjust(left=0.05)
    for i,lr in enumerate(LRS):
        for F_optim in F_optims:
            axarr[i].plot(dic[F_optim+lr+name],label=F_optim)
            axarr[i].set_title('{}-{}'.format(name,lr))
        axarr[i].legend()
    plt.savefig('{}'.format(name))
