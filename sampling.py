import os, sys, time
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def sampling(x, y, num=1000):
    index_list = list(range(x.shape[0]))
    prob = np.random.normal(loc=np.mean(index_list), scale=0.01, size=x.shape[0])
    prob = prob / np.sum(prob)
    index = np.random.choice(index_list, size=num, p=prob)
    return x[index], y[index]

def func(x):
    return np.sin(2 * np.pi * x)

def load(low=0, high=1., n=10, std=1., part=False):
    x = np.random.uniform(low, high, n)
    if part:
        t = func(x) + .2
    else:
        t = func(x) + np.random.normal(scale=std, size=n)
    return x, t

def main():
    # Data Load
    x, y = load(low=-1.5, high=1.5, n=5000, std=0.1)
    x_part, y_part = load(low=0, high=0.5, n=200, std=0, part=True)
    x_all, y_all = np.concatenate([x, x_part]), np.concatenate([y, y_part])
    x_sample, y_sample = sampling(x_all, y_all)

    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=3, color="#000000")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_part, y=y_part, mode='markers', marker=dict(size=3, color="#ff0000")), row=1, col=1)

    fig.add_trace(go.Scatter(x=x_all, y=y_all, mode='markers', marker=dict(size=3, color="#4169e1")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_sample, y=y_sample, mode='markers', marker=dict(size=3, color="#ff0000")), row=2, col=1)

    # Add range slider
    #fig.update_layout(xaxis=go.layout.XAxis(rangeslider=dict(visible=True)))
    fig.show()
    return

if __name__ == "__main__":
    main()