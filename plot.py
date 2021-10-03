import matplotlib.pyplot as plt
import numpy as np

def plot(train_log, valid_log, test_log, name, steps):
    
    epoch = [steps*i for i in range(len(test_log))]

    # hits at k graph
    x1 = epoch
    y1 = train_log
    y2 = valid_log
    y3 = test_log
    if train_log != None:
        plt.plot(x1, y1, label = "train log")
    plt.plot(x1, y2, label = "valid log")
    plt.plot(x1, y3, label = "test log")
    # naming the x axis
    plt.xlabel('x - epoch')
    # naming the y axis
    plt.ylabel('y - value')
    # giving a title to my graph
    plt.title(name)
    # show a legend on the plot
    plt.legend()
    plt.savefig(name+'.png')
    plt.clf()


def plot2(log, name, steps):
    
    epoch = [steps*i for i in range(len(log))]
    
    x1 = epoch
    y1 = log
    plt.plot(x1, y1, label = "log")
    # naming the x axis
    plt.xlabel('x - epoch')
    # naming the y axis
    plt.ylabel('y - value')
    # giving a title to my graph
    plt.title(name)
    # show a legend on the plot
    plt.legend()
    plt.savefig(name+'.png')
    plt.clf()
