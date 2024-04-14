import numpy as np
import matplotlib.pyplot as plt
def numerical_derivative(f,timestamps,timestamps_delta):
    f=np.array(f)
    timestamps=np.array(timestamps)
    timestamps_delta=np.array(timestamps_delta)
    x = np.arange(len(f))
    num_der = np.zeros(len(f))
    num_der_indices = np.arange(1, len(f)-1)
    for i in num_der_indices:
        num_der[i] = (f[i + 1] - f[i - 1]) / (timestamps_delta[i]+timestamps_delta[i+1])
    return num_der