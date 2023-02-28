import os
import matplotlib.pyplot as plt
Fedavg=[]
Fedavgm=[]
FedCnm=[]
Fedprox=[]
scaffold=[]
FedAdam=[]
FedAdam1=[]
logdir="./logs/"
with open(os.path.join(logdir, "fedavg_02-23_12-16.log"), 'r+') as f:
    while(1):
        str=f.readline()
        if not str:
             break
        if str[24:41:] == "Global Model Test":
            Fedavg.append(float(str[52:60:]))
with open(os.path.join(logdir, "fedcnm_02-23_20-56.log"), 'r+') as f:
    while (1):
        str = f.readline()
        if not str:
            break
        if str[24:41:] == "Global Model Test":
            FedCnm.append(float(str[52:60:]))
with open(os.path.join(logdir, "fedprox_02-24_05-54.log"), 'r+') as f:
    while (1):
        str = f.readline()
        if not str:
            break
        if str[24:41:] == "Global Model Test":
            Fedavgm.append(float(str[52:60:]))
with open(os.path.join(logdir, "fedavgm_02-24_16-06.log"), 'r+') as f:
    while 1:
        str = f.readline()
        if not str:
            break
        if str[24:41:] == "Global Model Test":# state of the art
            FedAdam.append(float(str[52:60:]))
with open(os.path.join(logdir, "fedadam_02-25_00-48.log"), 'r+') as f:
    while(1):
        str=f.readline()
        if not str:
            break
        if str[24:41:] == "Global Model Test":
            Fedprox.append(float(str[52:60:]))
with open(os.path.join(logdir, "fedadc_02-26_14-56.log"), 'r+') as f:
    while(1):
        str=f.readline()
        if not str:
             break
        if str[24:41:]=="Global Model Test":
            scaffold.append(float(str[52:60:]))
# with open(os.path.join('./logs/', "domo_11-21_17-06.log"), 'r+') as f:
#     while 1:
#         str = f.readline()
#         if not str:
#             break
#         if str[24:41:] == "Global Model Test":
#             FedAdam1.append(float(str[52:60:]))
print(Fedavg[-5::],len(Fedavg))
print(FedCnm[-5::],len(FedCnm))
print(Fedavgm[-5::],len(Fedavgm))
print(FedAdam[-5::],len(FedAdam))
print(Fedprox[-5::],len(Fedprox))
print(scaffold[-5::],len(scaffold))
print(FedAdam1[-5::],len(FedAdam1))
plt.plot(Fedavg[::],label='fedavgsm')
plt.plot(FedCnm[::],label='fedavg')
plt.plot(Fedavgm[::],label='fedavglm')
plt.plot(FedAdam[::],label='fedavglsm')
plt.plot(Fedprox[::],label='fedcnm')
plt.plot(scaffold[::],label='α=0.99')
plt.plot(FedAdam1[::],label='Fedadam1')


plt.title('CIFAR100 lr=0.1')
plt.grid(True)
plt.axis('tight')
plt.legend(loc=0)
plt.xlabel('Communication Rounds')
plt.ylabel('Test Accuracy')
plt.show()