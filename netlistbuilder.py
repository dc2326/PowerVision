from datetime import datetime
from PyLTSpice import SimCommander, RawRead
import os
import matplotlib.pyplot as plt

class NetList:

    # Make a new Netlist and name it using the date and time
    def __init__(self, fileName = datetime.now().strftime("%d%m%Y_%H%M%S")):
        self.netlist = fileName

        # Variable to keep track of number of extra drivers automatically added
        self.driver_node_count = 1

        file = open(self.netlist+".net","w")
        file.close()
    
    # Generate the netlist from matrix representation of circuit
    # con_matrix: connection matrix (list)
    # dev_matrix: device matrix (list)
    #
    # Connection Matrix format:
    #   Device 1: {1 0 2 ... }
    #   Device 2: {0 2 0 ... }
    #   Device 3: {2 1 0 ... }
    #   Device 4: {0 3 1 ... }
    #     ...   : {   ...    }
    #      Nodes:  0 1 2 ... --> Node 0 = GND, Node len[]-1 = Pin to be driven
    #
    # Device Matrix format:
    # {V, R, M, R, R, C, ...}
    # Use device matrix to figure out what devices 1, 2, 3, etc.
    # coorespond to.
    def generate(self, con_matrix, dev_matrix):

        # List of netlist non-source components
        comp = []

        # List of netlist source components
        sources = []

        # List of generated drivers (for driving unconnected MOSFET gates)
        drivers = []

        # Dictionary for keeping track of number of each component
        self.components = {}

        # Flag set if mosfet or diode model needs to be added
        self.add_mosfet = False
        self.add_diode = False

        # Get total number of devices
        total_devices = len(con_matrix)

        if (total_devices == 0):
            raise Exception("Empty connection matrix")

        # Get total number of nodes
        total_nodes = len(con_matrix[0])

        # Loop over all devices of connection matrix
        for i in range(total_devices):

            # Get device name/type first
            x = dev_matrix[i]

            # Device numbering
            # If first time seeing device
            if x not in self.components.keys():
                self.components[x] = 0
            
            # Increment component count, add it to running command list
            self.components[x] = self.components[x] + 1
            temp = x + str(self.components[x]) + ' '

            # Make the device connections
            j = 0
            while True:
                # Increment pin counter
                j = j + 1

                # Use try/except because components may have a varying number of pins
                try:
                    node = con_matrix[i].index(j)
                    
                    # Corner case: MOSFET bulk connection
                    if (j == 3 and x == "M"):
                        node = str(node) + " " + str(node)

                    # If a pin needs to be driven, chances are it's a MOSFET, so add appropriate driver
                    if (node == total_nodes-1):
                        node, output = self.makeDriver(con_matrix[i].index(j+1))
                        drivers.append(output)

                    temp = temp + str(node) + ' '
                except ValueError:
                    break
        
            # Check if there are diodes or mosfets
            if (x == 'M'):
                self.add_mosfet = True
                temp = temp + "mosfet"  
            elif (x == 'D'):
                self.add_diode = True
                temp = temp + "diode"
            else:
                # FIXME: Add component values
                if (x == 'V'):
                    temp = temp + "5"
                elif (x == 'R'):
                    temp = temp + "10"
                elif (x == 'L'):
                    temp = temp + "10u"
                elif (x == 'C'):
                    temp = temp + "100u"

            # Determine if source. 0 means no, 1 means yes
            if ((x == 'V') or (x == 'A')):
                sources.append(temp)
            else:
                comp.append(temp)

        # Write to cmds to file
        self.writeNet(comp, sources, drivers)

    # Creates and returns a PWM voltage node
    def makeDriver(self, ref, delay="0", offtime="3u", period="10u"):
        # Keeps track of the number of driver nodes
        d_count = self.driver_node_count
        self.driver_node_count += 1

        # Device numbering
        # If first time seeing Vdrv
        if "Vdrv" not in self.components.keys():
            self.components["Vdrv"] = 0
        
        # Increment component count, add it to running command list
        self.components["Vdrv"] = self.components["Vdrv"] + 1

        # Create pulse command
        pulse = "PULSE(20 0 {Tdelay} 0 0 {Ton} {Tperiod})".format\
            (Tdelay = delay, Ton = offtime, Tperiod = period)

        # Assemble command
        stub = "Vdrv" + str(self.components["Vdrv"]) + " p" + str(d_count) + " " + str(ref) + " " + pulse

        return "p"+str(d_count), stub
    
    # Write the cmds to the netlist
    def writeNet(self, comp, sources, drivers):
        n_file = open(self.netlist+".net", 'w')

        len_comp = len(comp)
        len_sources = len(sources)
        len_drivers = len(drivers)

        # Print a little comment
        print(self.writeNewSection("Netlist for "+self.netlist+".net"), file=n_file)

        # Loop through sources list and write to .net file
        if (len_sources > 0):
            print(self.writeNewSection("Sources:", True), file=n_file)
            for x in range(len_sources):
                print(sources[x], file=n_file)

        # Loop through component list and write to .net file
        if (len_comp > 0):
            print(self.writeNewSection("Components:", True), file=n_file)
            for x in range(len_comp):
                print(comp[x], file=n_file)

        # Loop through driver list and write to .net file
        if (len_drivers > 0):
            print(self.writeNewSection("Drivers:", True), file=n_file)
            for x in range(len_drivers):
                print(drivers[x], file=n_file)

        # Add MOSFET and Diode models if necessary
        if ((self.add_mosfet or self.add_diode) == True):
            print(self.writeNewSection("Models:", True), file=n_file)
            if (self.add_mosfet == True):
                print(".model mosfet NMOS(Kp=60 Vto=4.5)", file=n_file)
            if (self.add_diode == True):
                print(".model diode D", file=n_file)


        # Simulation cmd
        print("\n.tran 5m", file=n_file)

        # End the netlist
        print("\n.end", file=n_file)

        # Close the file
        n_file.close()

    # Creating a new section in netlist
    def writeNewSection(self, comment="", newline=False):
        string = ""
        # Print a newline
        if (newline == True):
            string = string + "\n"
        
        # Print a line comment
        string = string + "* " + comment
        
        return string

    # Run LTSpice simulation
    def run(self):
        # Get current working directory
        meAbsPath = os.path.dirname(os.path.realpath(__file__))
        
        # Set up LTSpice and run simulation
        LTC = SimCommander(meAbsPath + "/" + self.netlist + ".net")  
        LTC.run()
        LTC.wait_completion()

        # Parse the rawfile beforehand and save it
        print(self.netlist+"_1.raw")
        self.rawfile = RawRead(self.netlist + "_1.raw")

        # Create empty list to store measurement data
        self.traces = []

    # FIXME: Grab the voltage or current trace for a certain component
    # def grab(self, measurement, device):
    #     self.traces.append(self.rawfile.get_trace())

    # Plot all the values grabbed
    def plot(self):
        t = self.rawfile.get_trace('time')
        steps = self.rawfile.get_steps()

        self.traces=['V(4)', 'V(1)']

        for step in range(len(steps)):
            for mtr in range(len(self.traces)):
                plt.plot(t.get_wave(step), self.rawfile.get_trace(self.traces[mtr]).get_wave(step), label=self.traces[mtr])

        plt.legend()
        plt.show()
