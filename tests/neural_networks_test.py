#-*-coding:utf-8-*- 
import unittest as ut
from neural_networks import *

class TestNeuralNetworksMethods(ut.TestCase):
    def test_sigmoid_activation_function(self):
        self.assertEqual(0.00033535, round(sigmoid_activation_function(-8), 8))
        self.assertEqual(0.047426, round(sigmoid_activation_function(-3), 6))
    def test_Neuron_input(self):
        n = Neuron(input_activation_function, 0.5)
        self.assertEqual(0.1, round(n.input(0.6), 1))
        self.assertFalse(0, n.input(0.3))
        self.assertFalse(0, n.input(0.5))
    def test_Neuron_active(self):
        n = Neuron(input_activation_function, 0.5)
        ns = [Neuron(), Neuron(), Neuron()] # 下层神经元
        for nt in ns:
            NMConnection(nt, n, 0.1)
        inputs = [6, 3, 5]
        self.assertEqual(0.9, round(n.active(inputs), 1))
    def test_NeualNetworks_str(self):
        snm = SingleHidenLayerNM(2, 3, 2)
        print()
        print(snm)
        self.assertIsNotNone(str(snm))