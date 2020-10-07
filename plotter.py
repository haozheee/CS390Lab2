import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
datasets = ['mnist_d', "mnist_f", "cifar_10", "cifar_100_f", "cifar_100_c"]
#accuracy = [96.73, 79.32, 10.00, 1.00, 5.00]   # ANN
accuracy = [99.44, 91.49, 76.42, 38.06, 53.83]      # CNN
ax.bar(datasets,accuracy)
plt.legend(loc='best')
plt.show()
#plt.savefig('./ANN_Accuracy_Plot.pdf')         # ANN
plt.savefig('./CNN_Accuracy_Plot.pdf')        # CNN

