# Creates and simulates a basic boost converter

import numpy as np
import matplotlib.pyplot as plt
import sys

import PySpice
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.HighLevelElement import *
from PySpice.Unit import *

frequency = 100@u_kHz
period = frequency.period

logger = Logging.setup_logging()

# Creates the circuit
circuit = Circuit('Boost Converter')

# Defines models of components
# 1N4148 diode
circuit.model('MyDiode', 'D', IS=4.352@u_nA, RS=0.6458@u_Ohm, BV=110@u_V, IBV=0.0001@u_V, N=1.906)
# IRLZ44N MOSFET
circuit.model('MyMOSFET', 'NMOS', Kp=0.13, Vto=1.5)

# Adding components to the circuit
circuit.V('input', 1, circuit.gnd, 5@u_V)
# Input cap
#circuit.C(1, 'vin', circuit.gnd, 100@u_uF)
# 100uH inductor
circuit.L(1, 1, 'lout', 10@u_uH)
# M <name> <drain node> <gate node> <source node> <bulk/substrate node>
circuit.MOSFET(1, 'lout', 3, circuit.gnd, circuit.gnd, model='MyMOSFET')
# Diode wiring syntax: Dnnn anode cathode <model>
circuit.Diode(1, 'lout', 4, model='MyDiode')
# Output cap
circuit.C(2, 4, circuit.gnd, 100@u_uF)
# 1 ohm resistor load
circuit.R(1, 4, circuit.gnd, 10@u_Ohm)

circuit.PulseVoltageSource('PWM', 'gater', circuit.gnd, initial_value=0@u_V, pulsed_value=10@u_V, delay_time=0, rise_time=1e-15, fall_time=1e-15, pulse_width=0.5*1e-5, period=1e-5)
circuit.R(2, 'gater', 3, 0.1@u_Ohm)

# Diode wiring syntax: Dnnn anode cathode <model>
#circuit.Diode(1, 2, 3, model='MyDiode')

# Print Circuit netlist
print("The Circuit/Netlist:\n\n", circuit)

# Creates simulator object
simulator = circuit.simulator(temperature=25, nominal_temperature=25)

# Prints simulator details
#print("The Simulator:\n\n", simulator)

# Run analysis
analysis = simulator.transient(step_time=0.0001, end_time = 0.01)

# Print where analysis is stored
#print(analysis)

# Print node values manually
# print(analysis.nodes['in'])
#print("Node:", str(analysis["1"]), "Values:", np.array(analysis["1"]))
#print("Node:", str(analysis["4"]), "Values:", np.array(analysis["4"]))

fig = plt.figure()

plt.plot(np.array(analysis.time), np.array(analysis["3"]))
#plt.plot(np.array(analysis.time), np.array(analysis["4"]))
#plt.plot(np.array(analysis.time), np.array(analysis["1"]))
plt.xlabel("Time")
plt.ylabel("Output Voltage")
plt.show()

# print(str(analysis.nodes['1']))
# print(float(analysis.nodes['1']))

# print(str(analysis.nodes['4']))
# print(float(analysis.nodes['4']))

# Print formatted node values

exit()