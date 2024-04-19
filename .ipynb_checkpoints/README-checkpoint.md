# PowerVision

A neural net that reads a schematic, turns it into a SPICE netlist, and simulates it.

Sources:
https://towardsdatascience.com/remove-text-from-images-using-cv2-and-keras-ocr-24e7612ae4f4

https://medium.com/bitgrit-data-science-publication/building-an-image-classification-model-with-pytorch-from-scratch-f10452073212

https://pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/


Requirements:

1. PyLTSpice:

        pip install PyLTSpice

2. PyTorch: 

        See installation instructions here: https://pytorch.org/get-started/locally/        

3. OpenCV and other important libraries:

        pip install opencv-python numpy matplotlib

4. LTSpice:

        https://www.analog.com/en/resources/design-tools-and-calculators/ltspice-simulator.html

5. Virtual Environment:

        Recommended but not needed

        pip install virtualenv
        python -m venv powervision

        * To activate venv:
        source powervision/Scripts/activate

        * To deactivate venv:
        deactivate
