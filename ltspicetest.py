# Code from: https://www.youtube.com/watch?v=nv9xDpM30Ig&ab_channel=LandonBallard

from PyLTSpice import SimCommander, RawRead
import numpy as np
import os
import matplotlib.pyplot as plt

meAbsPath = os.path.dirname(os.path.realpath(__file__))
LTC = SimCommander(meAbsPath + "\\boostconverter.net")
LTC.run()
LTC.wait_completion()

#l = ltspice.Ltspice('test_1.raw')
l = RawRead("boostconverter_1.raw")
y1 = l.get_trace('V(4)')
x = l.get_trace('time')
steps = l.get_steps()

for step in range(len(steps)):
    plt.plot(x.get_wave(step), y1.get_wave(step), label="Vout")

plt.legend()
plt.show()

#print(l.get_trace_names())


