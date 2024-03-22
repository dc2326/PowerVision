# Code from: https://www.youtube.com/watch?v=0uAIrcrn-ww

# Creates and simulates a simple voltage divider

import numpy as np
import matplotlib.pyplot as plt
import sys

import PySpice
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

logger = Logging.setup_logging()

# Creates the circuit
circuit = Circuit('DC Sweep Test')

# Defines models of components
# 1N4148 diode
circuit.model('MyDiode', 'D', IS=4.352@u_nA, RS=0.6458@u_Ohm, BV=110@u_V, IBV=0.0001@u_V, N=1.906)

# Adding components to the circuit
circuit.V('input', 1, circuit.gnd, 10@u_V)
circuit.Diode(1, 1, 2, model='MyDiode')
circuit.R(1, 2, circuit.gnd, 1@u_kOhm)


# Print Circuit netlist
#print("The Circuit/Netlist:\n\n", circuit)

# Creates simulator object
simulator = circuit.simulator(temperature=25, nominal_temperature=25)

# Prints simulator details
#print("The Simulator:\n\n", simulator)

# Run analysis
analysis = simulator.dc(Vinput=slice(0, 5, 0.1))

# Print where analysis is stored
#print(analysis)

# Print node values manually
# print(analysis.nodes['in'])
# print(str(analysis.nodes['in']))
# print(float(analysis.nodes['in']))

# print(str(analysis.nodes['out']))
# print(float(analysis.nodes['out']))

# Print formatted node values
print("Node:", str(analysis["1"]), "Values:", np.array(analysis["1"]))
print("Node:", str(analysis["2"]), "Values:", np.array(analysis["2"]))

# Plot data!
fig = plt.figure()

plt.plot(np.array(analysis["1"]), np.array(analysis["2"]))
plt.xlabel("Input Voltage (Node 1)")
plt.ylabel("Output Voltage (Node 2)")

fig.savefig("Sim_Output.png", dpi=300)
plt.close(fig)

exit()





