# annie-capture-demo-python-app

This is a quick python demo application hacked using AMD NN inference engine.
## Steps:

1. **Convert net.caffemodel into NNIR model using the following command**
   ````
	    python caffe2nnir.py <net.caffeModel> <nnirOutputFolder> --input-dims n,c,h,w [--verbose 0|1]
   ````
2. **Compile NNIR model into OpenVX C code with CMakelists.txt for compiling and building inference library**
   ````
	    python nnir2openvx.py <nnirModelFolder> <nnirModelOutputFolder>
   ````
3. **Copy .so's into annieCapture**
 
 Copy the libannmodule.so,libannpython.so, & weights.bin built inside the resnet50-build folder into the annieCapture folder.

4. **Run ANNIE Capture**

 To run the capture make sure that you have a webcam connected and run the following command inside the annieCapture folder.

 ````
 python annieCapture.py 
 ````
## Script Inputs:
````
--image [input image file     - required]
--imagefolder [input Image Directory - required]
--capture [0/1] - (camera input - required)
--pm - [preprocess multiply(as per trained model) - required]
--pa - [preprocess add(as per trained model) - required]
--annpythonlib - [pythonlib - optional]
--weights - [weights.bin - optional]
--labels - [labelfile.txt - optional]
--hierarchy - [hierarchyfile - optional]
--resultsfolder - [output results directory - optional]
````
## Sample (for Resnet)
```
python AnnieCapture.py --capture [0/1] --pm 1 --pa 0 --annpythonlib <annpythonlib> --weights <weights.bin> --labels labelfile.txt --hierarchy <hierarchyfile>

python AnnieCapture.py --imagefolder <input image directory> --pm 1 --pa 0 --annpythonlib <annpythonlib> --weights <weights.bin> --labels labelfile.txt --hierarchy <hierarchyfile>

python AnnieCapture.py --image <input image file> --pm 1 --pa 0 --annpythonlib <annpythonlib> --weights <weights.bin> --labels labelfile.txt --hierarchy <hierarchyfile>
```
## Key Press Options when in capture mode:
Press these different keys to switch between modes (uses openCV)
1. **Keys '1' through 'n'** - Runs through a folder corresponding to number once and goes back to live mode (currently supports keys 1,2 folders)

2. **Key 'f'** - Runs through a folder until asked to change

3. **Key 'q'** - Quits from the program

4. **Key 'Space Bar'** - pauses the capture until space bar pressed again

## Key Press Options when in imagefolder mode:
Press these different keys to switch between modes (uses openCV)
 1. **Key 'c'** - Switches to camera capture mode until asked to change
 
 2. **Key 'q'** - Quits from the program
 
 3. **Key 'Space Bar'** - pauses the capture until space bar pressed again
