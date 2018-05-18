# annie-capture-demo-python-app

This is a quick python demo application hacked using AMD NN inference engine.
Steps:
1) Run caffe2Openvx tool to get compiled model runtime module
2) Modify annmodule.cpp and annmodule.h for python interface (Copy of those files are in the same folder for inception v4)
2) build libannmodule.so which will be linked to the python code for running inference

TODO:
AnnCapture_new.py links with the model compilers automatically generated python lib. (Currently not working)

Command: "python AnnieCapture.py --video null --labels labelfile.txt"
