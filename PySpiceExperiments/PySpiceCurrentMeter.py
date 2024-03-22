import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

circuit = Circuit('Wheatstone Bridge')

circuit.V('Input', 'vin', circuit.gnd, 10@u_V)
circuit.R(1, 'vin', 'a', 2@u_kOhm)
circuit.R(2, 'vin', 'b', 1@u_kOhm)
circuit.R(3, 'a', 'b', 2@u_kOhm)
circuit.R(4, 'b', circuit.gnd, 2@u_kOhm)
circuit.R(5, 'a', circuit.gnd, 1@u_kOhm)

for resistance in (circuit.R1, circuit.R2, circuit.R3, circuit.R4, circuit.R5):
    resistance.plus.add_current_probe(circuit)

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.operating_point()

# Gets branch current
for node in analysis.branches.values():
    print('Node {}: {:5.3f} A'.format(str(node), float(node)))

# Gets node voltages
for node in analysis.nodes.values():
    print('Node {}: {:5.3f} V'.format(str(node), float(node)))