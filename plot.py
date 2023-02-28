import sys
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
Fedavg=[]
Fedavg1=[]
Fedprox=[]
Fedprox1=[]
Fedadc=[]
Fedadc1=[]
FedAdam=[]
FedAdam1=[]
Fedavgm=[]
Fedavgm1=[]
FedCnm=[]
FedCnm1=[]

min1=0.27
round1=499
round2=999
logdir="./logs"
with open(os.path.join(logdir, "fedavg_02-21_11-05.log"), 'r+') as f:
    while(1):
        str=f.readline()
        if not str:
             break
        if str[24:41:] == "Global Model Test":
            Fedavg.append(float(str[52:60:]))
# with open(os.path.join(logdir, "fedadc_11-13_09-59.log"), 'r+') as f:
#     while (1):
#         str = f.readline()
#         if not str:
#             break
#         if str[24:41:] == "Global Model Test":
#             Fedadc.append(float(str[52:60:]))
# with open(os.path.join(logdir, "fedadam_12-07_05-25.log"), 'r+') as f:
#     while(1):
#         str=f.readline()
#         if not str:
#             break
#         if str[24:41:] == "Global Model Test":
#             FedAdam.append(float(str[52:60:]))
# with open(os.path.join(logdir, "fedavgm_09-11_22-46.log"), 'r+') as f:
#     while(1):
#         str=f.readline()
#         if not str:
#             break
#         if str[24:41:] == "Global Model Test":
#             Fedavgm.append(float(str[52:60:]))
# with open(os.path.join(logdir, "fedcnm_09-08_22-38.log"), 'r+') as f:
#     while(1):
#         str=f.readline()
#         if not str:
#             break
#         if str[24:41:] == "Global Model Test":
#             FedCnm.append(float(str[52:60:]))

Fedavg1.append(Fedavg[0])
Fedadc1.append(Fedadc[0])
# FedAdam1.append(FedAdam[0])
Fedavgm1.append(Fedavgm[0])
FedCnm1.append(FedCnm[0])

for i in range(round2):
    Fedavg1.append(Fedavg1[i]*0.9+Fedavg[i]*0.1)
    Fedadc1.append(Fedadc1[i]*0.9+Fedadc[i]*0.1)
    # FedAdam1.append(FedAdam1[i]*0.9+FedAdam[i]*0.1)
    Fedavgm1.append(Fedavgm1[i]*0.9+Fedavgm[i]*0.1)
    FedCnm1.append(FedCnm1[i]*0.9+FedCnm[i]*0.1)

# # print(Fedavg[-10::],len(Fedavg))
# # # print(Fedprox[-10::],len(Fedprox))
# # print(Fedadc[-10::],len(Fedadc))
# # print(FedAdam[-10::],len(FedAdam))
# # print(Fedavgm[-10::],len(Fedavgm))
# # print(FedCnm[-10::],len(FedCnm))
# base=0
# for i in range(round2):
#     if Fedavg1[i]>=min1:
#         j=i*2
#         print(i,i*2,1,'Fedavg')
#         base=j
#         break
# for i in range(round2):
#     if Fedadc1[i]>=min1:
#         j=i*3
#         print(i,j,j/base,'Fedadc')
#         break
# # for i in range(round2):
# #     if FedAdam1[i]>=min1:
# #         j=i*2
# #         print(i,j,j/base,'Fedadam')
# #         break
# for i in range(round2):
#     if Fedavgm1[i]>=min1:
#         j=i*2
#         print(i,j,j/base,'Fedavgm')
#         break
# for i in range(round2):
#     if FedCnm1[i]>=min1:
#         j=i*3
#         print(i,j,j/base,'Fedcnm')
#         break

print(Fedavg1[round1],Fedadc1[round1],Fedavgm1[round1],FedCnm1[round1])
print(Fedavg1[round2],Fedadc1[round2],Fedavgm1[round2],FedCnm1[round2])

plt.plot(Fedavg1[::],label='Fedavg',linestyle='--')
plt.plot(Fedadc1[::],label='Fedadc',linestyle='-.')
# plt.plot(FedAdam1[::],label='Fedadam')
plt.plot(Fedavgm1[::],label='Fedavgm')
plt.plot(FedCnm1[::],label='Fedcnm',linestyle=':')


# plt.plot(Fedavg[::],label='α=0')
# plt.plot(Fedadc[::],label='α=0.3')
# plt.plot(FedAdam[::],label='α=0.9')
# plt.plot(Fedavgm[::],label='α=0.95')
# plt.plot(FedCnm[::],label='α=0.99')

plt.title('CIFAR100,IID,200 Devices,5%')
plt.grid(True)
plt.axis('tight')
plt.legend(loc=0)
plt.xlabel('Communication Rounds')
plt.ylabel('Test Accuracy')
plt.show()