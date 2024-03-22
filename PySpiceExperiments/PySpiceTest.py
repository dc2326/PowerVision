# Code from: https://www.youtube.com/watch?v=0uAIrcrn-ww

# Creates and simulates a simple voltage divider

import numpy as np
import matplotlib as plt
import sys

import PySpice
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

def format_output(analysis):
    sim_res_dict = {}

    for node in analysis.nodes.values():
        data_label = "%s" % str(node)                       # extract node name
        sim_res_dict[data_label] = np.array(node)           # save node value/array of values

    return sim_res_dict

logger = Logging.setup_logging()

# Creates the circuit
circuit = Circuit('Voltage Divider')

# Adding components to the circuit
circuit.V('input', 'in', circuit.gnd, 10@u_V)
circuit.R(1, 'in', 'out', 9@u_kOhm)
circuit.R(2, 'out', circuit.gnd, 1@u_kOhm)

# Print Circuit netlist
print("The Circuit/Netlist:\n\n", circuit)

# Creates simulator object
simulator = circuit.simulator(temperature=25, nominal_temperature=25)

# Prints simulator details
print("The Simulator:\n\n", simulator)

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





