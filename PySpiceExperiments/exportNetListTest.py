from fileHandling import NetListFile

import numpy as np
import matplotlib as plt
import sys

import PySpice
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from PySpice.Spice.NgSpice.Shared import NgSpiceShared
ngspice = NgSpiceShared.new_instance()

logger = Logging.setup_logging()

f = NetListFile()

# Creates the circuit
circuit = Circuit('Voltage Divider')

# Adding components to the circuit
circuit.V('input', 'in', circuit.gnd, 10@u_V)
circuit.R(1, 'in', 'out', 9@u_kOhm)
circuit.R(2, 'out', circuit.gnd, 1@u_kOhm)

# Creates simulator object
simulator = circuit.simulator(temperature=25, nominal_temperature=25)

# Prints simulator details
f.write(simulator)

exit()