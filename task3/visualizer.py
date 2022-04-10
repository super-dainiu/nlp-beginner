import numpy as np
import matplotlib.pyplot as plt

x_range = np.arange(0, 2, 0.2) + 0.2
print(x_range)
train_loss_p, train_acc_p, dev_acc_p = np.load('train_loss_parikh.npy'), np.load('train_acc_parikh.npy'), np.load('dev_acc_parikh.npy')
train_loss_e, train_acc_e, dev_acc_e = np.load('train_loss_esim.npy'), np.load('train_acc_esim.npy'), np.load('dev_acc_esim.npy')
plt.plot(x_range, train_loss_p)
plt.plot(x_range, train_acc_p)
plt.plot(x_range, dev_acc_p)
plt.plot(x_range, train_loss_e)
plt.plot(x_range, train_acc_e)
plt.plot(x_range, dev_acc_e)
plt.title("model_evaluation")
plt.xlabel('epoch')
plt.legend(['train_loss parikh', 'train_acc parikh', 'dev_acc parikh', 'train_loss esim',
            'train_acc esim', 'dev_acc esim'])
plt.show()