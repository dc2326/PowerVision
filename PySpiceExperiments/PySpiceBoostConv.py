# Creates and simulates a basic boost converter

# Required PySpice Libs
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

from PySpice.Spice.Netlist import Circuit
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Spice.HighLevelElement import *
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Unit import *

# Numpy stuff
import numpy as np
import matplotlib.pyplot as plt

libraries_path = find_libraries()
spice_library = SpiceLibrary(libraries_path)

# Creates the circuit
circuit = Circuit('Boost Converter')

# # Defines models of components
# # 1N4148 diode

# # MOSFET
# #.model BSZ0905NS VDMOS(Rg=3 Vto=2.3 Rd=3.69m Rs=2.12m Rb=5.17m Kp=110.4 Lambda=0.07 Cgdmin=14p Cgdmax=0.24n A=0.6 Cgs=0.52n Cjo=0.73n M=0.75 Is=5.8p VJ=2.5 N=1.12 TT=3n mfg=Infineon Vds=30 Ron=9m Qg=4n)
#circuit.model('MyMOSFET', 'NMOS', L=100E-6, W=200E-6, Kp=0.13, Vto=2.475)
circuit.model('MyMOSFET', 'NMOS', L=100E-6, W=200E-6, Kp=1, Vto=2.475)

#circuit.include(spice_library['1N5822']) # Schottky diode
#circuit.include(spice_library['irf150'])

#circuit.model('MyDiode', 'D', IS=76.9E-12, RS=42.0E-3, BV=100, IBV=5E-6, CJO=39.8E-12, M=0.333, N=1.45, TT=4.32E-6)
circuit.model('MyDiode', 'D', IS=4.352@u_nA, RS=1@u_mOhm, BV=510@u_V, IBV=0.0001@u_V, N=1.906)

# Adding components to the circuit
circuit.V('input', 1, circuit.gnd, 12@u_V)
# Input cap
#circuit.C(1, 'vin', circuit.gnd, 100@u_uF)
# 10uH inductor
circuit.L(1, 1, 2, 10E-6)

# Either use MOSFET or Ideal Switch, though Ideal switch creates odd waveforms
# M <name> <drain node> <gate node> <source node> <bulk/substrate node>
#circuit.X('Q', 'irf150', 2, 3, circuit.gnd)
circuit.MOSFET(1, 2, 3, circuit.gnd, circuit.gnd, model='MyMOSFET')

#circuit.model('switch', 'SW', Ron=1@u_mOhm, Roff=1@u_GOhm)

# Diode wiring syntax: Dnnn anode cathode <model>
circuit.D(1, 2, 4, model='MyDiode')
#circuit.X('D', '1N5822', 2, 4)
# Output cap
circuit.C(2, 4, circuit.gnd, 100@u_uF)
# 1 ohm resistor load
circuit.R(1, 4, circuit.gnd, 5@u_Ohm)

circuit.PulseVoltageSource('PWM', 3, circuit.gnd, initial_value=0@u_V, pulsed_value=10@u_GV, delay_time=0, rise_time=1@u_ps, fall_time=1@u_ps, pulse_width=0.5@u_us, period=2@u_us)

#circuit.VoltageControlledSwitch(1, 'lout', circuit.gnd, 'gater', circuit.gnd, model='switch')

# Diode wiring syntax: Dnnn anode cathode <model>
#circuit.Diode(1, 2, 3, model='MyDiode')

# Print Circuit netlist
print("The Circuit/Netlist:\n\n", circuit)

# Creates simulator object
simulator = circuit.simulator(temperature=25, nominal_temperature=25)

# Prints simulator details
#print("The Simulator:\n\n", simulator)

# Run analysis
analysis = simulator.transient(step_time=1, end_time = 1)

# Print where analysis is stored
#print(analysis)

# Print node values manually
# print(analysis.nodes['in'])
#print("Node:", str(analysis["1"]), "Values:", np.array(analysis["1"]))
#print("Node:", str(analysis["4"]), "Values:", np.array(analysis["4"]))

fig = plt.figure()

#plt.plot(np.array(analysis.time), np.array(analysis["2"]))
plt.plot(np.array(analysis.time), np.array(analysis["4"]))
plt.plot(np.array(analysis.time), np.array(analysis["1"]))
plt.xlabel("Time")
plt.ylabel("Output Voltage")
plt.show()

# print(str(analysis.nodes['1']))
# print(float(analysis.nodes['1']))

# print(str(analysis.nodes['4']))
# print(float(analysis.nodes['4']))

# Print formatted node values

exit()