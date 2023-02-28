import matplotlib.pyplot as plt
import  numpy as np

size = 3

x = np.arange(size)

total_width , n = 0.6 , 4
width = total_width / n
list1=[91.34,90.33,83.50]
list2=[82.49,67.42,33.84]
list3=[79.87,88.20,15.61]
list4=[93.63,92.09,89.88]
x =x -(total_width - width)/2
print(x)
plt.rcParams['font.serif']=['Times New Roman']

plt.bar(x,list1,width=width,label="FedAvg",)