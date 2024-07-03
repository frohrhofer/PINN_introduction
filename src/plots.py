import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_input_data(X_IC, u_IC, X_BC, u_BC):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_IC[:,1], X_IC[:,0], u_IC[:,0], label='IC')
    ax.scatter(X_BC[:,1], X_BC[:,0], u_BC[:,0], label='BC')
    
    ax.legend(frameon=False)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$u(x,t)$')
    
    plt.tight_layout()
    plt.show()
    

def plot_prediction(model):
    
    N = 1000
    
    x_ticks = np.linspace(0, 1, N)
    t_ticks = np.linspace(0, 0.5, N)
    xx, tt = np.meshgrid(x_ticks, t_ticks)
    X = tf.convert_to_tensor(
                    np.vstack((xx.flatten(), tt.flatten())).T,
                    dtype=tf.float32
                    )
    
    u_pred = model(X)
    u_pred = u_pred.numpy().reshape(N, N)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(tt, xx, u_pred, facecolors=cm.jet(u_pred/u_pred.max()), lw=0)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$u(x,t)$')
    
    plt.tight_layout()
    plt.show()
    