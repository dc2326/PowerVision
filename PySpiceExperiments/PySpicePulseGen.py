import numpy as np
import matplotlib.pyplot as plt

import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.HighLevelElement import *
from PySpice.Unit import *

# Necessary
logger = Logging.setup_logging()

# Circuit Name
circuit = Circuit('Pulse Generator + Switch')

# Model of SW
#circuit.model('switch', 'SW', Ron=1@u_uOhm, Roff=1@u_GOhm)
circuit.model('MyMOSFET', 'NMOS', L=100E-6, W=200E-6, Kp=1, Vto=2.475)

# Pulsed voltage source
circuit.PulseVoltageSource('PWM', 3, circuit.gnd, \
                            initial_value=0@u_V, \
                            pulsed_value=10@u_V, \
                            delay_time=0, \
                            rise_time=1@u_ps, \
                            fall_time=1@u_ps, \
                            pulse_width=1@u_us, \
                            period=2@u_us)

# DC source
circuit.V(1, 1, circuit.gnd, 10@u_V)

circuit.R(1, 1, 2, 100@u_Ohm)

circuit.MOSFET(1, 2, 3, circuit.gnd, circuit.gnd, model='MyMOSFET')

# AC source
#circuit.SinusoidalVoltageSource(1, 10, 11, amplitude=1@u_V, frequency=60@u_Hz)



#circuit.VCS('Switch', 2, circuit.gnd, 3, circuit.gnd, model='switch')
#circuit.R(1, 2, circuit.gnd, 1@u_kOhm)

# Simulator essential things ----------------------------------------
simulator = circuit.simulator(temperature=25, nominal_temperature=25)

analysis = simulator.transient(step_time=0.0001@u_s, end_time = 0.00011@u_s)

fig = plt.figure()

plt.plot(np.array(analysis.time), np.array(analysis["2"]))
plt.plot(np.array(analysis.time), np.array(analysis.branches[('v1')]))
plt.xlabel("Time")
plt.ylabel("Output Voltage")
plt.show()

exit()

