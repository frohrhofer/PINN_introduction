import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm


def plot_datasets(X_IC, u_IC, X_BC, u_BC, X_col):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_IC[:,1], X_IC[:,0], u_IC[:,0], label='IC')
    ax.scatter(X_BC[:,1], X_BC[:,0], u_BC[:,0], label='BC')
    ax.scatter(X_col[:,1], X_col[:,0], 0, label='col')
    
    ax.legend(frameon=False)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$u(x,t)$')
    
    plt.tight_layout()
    plt.show()


def plot_learning_curves(log_data, log_physics):
    
    epochs = range(len(log_data))

    fig, ax = plt.subplots()

    ax.plot(epochs, log_data, label='loss_data')
    ax.plot(epochs, log_physics, label='loss_physics')

    ax.set_yscale('log')
    ax.legend(frameon=False)
    ax.set_xlabel('Training Epoch')

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
    