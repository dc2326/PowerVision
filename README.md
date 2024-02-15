# PowerVision

A neural net that reads a schematic and turns it into a SPICE netlist.

Requirementsngspic:

1. Need to install PySpice using:

        pip install PySpice
        pyspice-post-installation --install-ngspice-dll

2. Verify PySpice works with this command:

        pyspice-post-installation --check-install
