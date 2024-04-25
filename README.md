# PowerVision

![alt text](POWERVISION.png)

A piece of software that reads a schematic, turns it into a SPICE netlist, and simulates it.

Currently this software is optimized for planar circuits that do not have transformers.


# Requirements:

1. PyLTSpice:

        pip install PyLTSpice

2. TensorFlow: 

        // If your computer has an NVIDIA GPU:
        pip install tensorflow[and-cuda]

        // If not:
        pip install tensorflow

3. OpenCV and other important libraries:

        pip install opencv-python numpy matplotlib
        pip install scikit-image
        pip install imutils

4. LTSpice:

        https://www.analog.com/en/resources/design-tools-and-calculators/ltspice-simulator.html

5. Virtual Environment:

        pip install virtualenv
        python -m venv powervision

        * To activate venv:
        source powervision/Scripts/activate

        * To deactivate venv:
        deactivate


# How to use PowerVision

0. Clone this repo.

1. Create a Virtual Environment (venv) called powervision by following the steps in the previous section. This is important as this will isolate this software from the rest of your computer in case there are any clashes between libraries, and on Windows machines it allows you to install Tensorflow without facing the "long path" issue.

2. Install all required libraries and dependencies.

3. Upload the circuit you want to analyze, making sure that it is in the same directory as your powervision venv folder.

4. Run powervision.py using:

        python powervision.py

5. Follow the instructions, try it out with the provided examples in the example folder.


# How it works:

1. Uses image processing libraries and OpenCV on an image of a schematic in order to narrow down the search for components.

2. Crop the components found in the schematic and note their locations

3. Run images a trained CNN to identify the type of component.

4. Take each cropped image and do some more image processing to determine the orientation of the components

5. Generate a netlist matrix.

6. Run the matrix through a netlist generator, which also automatically adds gate drivers for MOSFETs

7. Calls LTSpice on the netlist, reads the RAW file and plots the data using matplotlib.
