# annie-capture-demo-python-app

This is a quick python demo application hacked using AMD NN inference engine.
Steps:
1) Run caffe2nnir and nnir2openvx which will compile the annmodule python library for inference
2) give the path of libannpython.so to the capture module to build 

TODO:

Command: "python AnnieCapture.py --capture [0/1] --annpythonlib <annpythonlib> --weights <weights.bin> --labels labelfile.txt --hierarchy <hierarchyfile>"
