import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')


x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
w = 200 #weight of the regression
b = 100 #bias of the regression

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m) #creates an empty array of predictions of y
    for i in range(m): #fuels the precedent array
        f_wb[i] = w * x[i] + b
        
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.show()
