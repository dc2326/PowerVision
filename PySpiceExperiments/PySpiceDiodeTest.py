# Code from: https://www.youtube.com/watch?v=znaERIx1tL8

# Creates and simulates a diode circuit with resistors

import numpy as np
import matplotlib as plt
import sys

import PySpice
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *

logger = Logging.setup_logging()

def format_output(analysis):
    sim_res_dict = {}

    for node in analysis.nodes.values():
        data_label = "%s" % str(node)                       # extract node name
        sim_res_dict[data_label] = np.array(node)           # save node value/array of values

    return sim_res_dict

class mySubCir(SubCircuit):
    __nodes__ = ('t_in', 't_out')
    def __init__(self, name, r=1@u_kOhm):

        SubCircuit.__init__(self, name, *self.__nodes__)

        self.R(1, 't_in', 't_out', r)
        self.Diode(1, 't_in', 't_out', model="MyDiode")


# Creates the circuit
circuit = Circuit('Diode Circuit')

# Defines models of components
# 1N4148 diode
circuit.model('MyDiode', 'D', IS=4.352@u_nA, RS=0.6458@u_Ohm, BV=110@u_V, IBV=0.0001@u_V, N=1.906)


# Adding components to the circuit
circuit.V('input', 1, circuit.gnd, 10@u_V)
circuit.R(1, 1, 2, 9@u_kOhm)

# Diode wiring syntax: Dnnn anode cathode <model>
circuit.Diode(1, 2, 3, model='MyDiode')

circuit.subcircuit(mySubCir('sub1',r=1@u_kOhm))
circuit.X(1, 'sub1', 3, circuit.gnd)

# Print Circuit netlist
#print("The Circuit/Netlist:\n\n", circuit)

# Creates simulator object
simulator = circuit.simulator(temperature=25, nominal_temperature=25)

# Prints simulator details
#print("The Simulator:\n\n", simulator)

# Run analysis
analysis = simulator.operating_point()

# Print where analysis is stored
#print(analysis)

# Print node values manually
# print(analysis.nodes['in'])
# print(str(analysis.nodes['in']))
# print(float(analysis.nodes['in']))

# print(str(analysis.nodes['out']))
# print(float(analysis.nodes['out']))

# Print formatted node values
out_dict = format_output(analysis)
print(out_dict)

exit()