import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import spline

epochs=np.arange(5000,50001,5000)
mae_loss=[0.500225365,0.000221096,0.000060971,0.000060323,0.000059905,0.000059579,0.000059274,
          0.000058972,0.000058697,0.000058476]
mse_loss=[0.135419831,0.018331185,0.002481434,0.000335913,0.000045486,0.000006180,0.000000867,
          0.000000147,0.000000042,0.000000042]
rmse_loss=[0.500225306,0.000293739,0.000126985,0.000121944,0.000119484,0.000117791,0.000116400,
           0.000115198,0.000114148,0.000113228]

def plotting(loss_type,title,color):
    xnew = np.linspace(epochs.min(),epochs.max(),300) #300 represents number of points to make between min and max
    loss = spline(epochs,loss_type,xnew)
    plt.plot(xnew, loss, color)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss value')
    plt.show()


mae = plotting(mae_loss,'Mean Absolute Error','g')
mse = plotting(mse_loss,'Mean Square Error','m')
rmse = plotting(rmse_loss,'Root Mean Square Error','c')

# logarithmic y-scale plot
plt.semilogy(epochs, mse_loss, 'r', label='MSE')
plt.semilogy(epochs, rmse_loss, 'g', label='RMSE')
plt.semilogy(epochs, mae_loss, 'b', label='MAE')
plt.title('Loss functions result, y-Logarithmic')
plt.xlabel('Epochs')
plt.ylabel('Log loss value')
plt.legend()
plt.show()
