from network import *
from linear_layer import *
from activation_layer import *
from activation_functions import *
from loss import *


a_b = np.array(
    [
        [[0, 0]],
        [[0, 1]],
        [[1, 0]],
        [[1, 1]],
    ]
)
a_xor_b = np.array(
    [
        [[0]],
        [[1]],
        [[1]],
        [[0]],
    ]
)

net = Network()
net.set_loss_function(bce)
net.add_layer(LinearLayer(2, 3))
net.add_layer(ActivationLayer(sigmoid))
net.add_layer(LinearLayer(3, 1))
net.add_layer(ActivationLayer(sigmoid))

net.train(a_b, a_xor_b, 4.7, 1000, 2)

print(net.predict(np.array([[1, 1]])))
