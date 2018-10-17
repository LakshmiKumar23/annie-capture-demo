"""
Demo:Classification Live Capture and folder modes
"""

import sys,os,time,csv,getopt,cv2,argparse
import numpy, ctypes, array
import numpy as np
import ntpath
import scipy.misc

from datetime import datetime
from ctypes import cdll, c_char_p
from skimage.transform import resize
from numpy.ctypeslib import ndpointer

from PIL import Image

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class AnnAPI:
	def __init__(self,library):
		self.lib = ctypes.cdll.LoadLibrary(library)
		self.annQueryInference = self.lib.annQueryInference
		self.annQueryInference.restype = ctypes.c_char_p
		self.annQueryInference.argtypes = []
		self.annCreateInference = self.lib.annCreateInference
		self.annCreateInference.restype = ctypes.c_void_p
		self.annCreateInference.argtypes = [ctypes.c_char_p]
		self.annReleaseInference = self.lib.annReleaseInference
		self.annReleaseInference.restype = ctypes.c_int
		self.annReleaseInference.argtypes = [ctypes.c_void_p]
		self.annCopyToInferenceInput = self.lib.annCopyToInferenceInput
		self.annCopyToInferenceInput.restype = ctypes.c_int
		self.annCopyToInferenceInput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_bool]
		self.annCopyFromInferenceOutput = self.lib.annCopyFromInferenceOutput
		self.annCopyFromInferenceOutput.restype = ctypes.c_int
		self.annCopyFromInferenceOutput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t]
		self.annRunInference = self.lib.annRunInference
		self.annRunInference.restype = ctypes.c_int
		self.annRunInference.argtypes = [ctypes.c_void_p, ctypes.c_int]
		print('OK: AnnAPI found "' + self.annQueryInference().decode("utf-8") + '" as configuration in ' + library)

def PreprocessImage(img, dim, pmul, padd):
	imgw = img.shape[1]
	imgh = img.shape[0]
	imgb = np.empty((dim[0], dim[1], 3))    #for inception v4
	imgb.fill(1.0)
	if imgh*dim[0] > imgw*dim[1]:
		neww = int(imgw * dim[1] / imgh)
		newh = dim[1]
	else:
		newh = int(imgh * dim[0] / imgw)
		neww = dim[0]
	offx = int((dim[0] - neww)/2)
	offy = int((dim[1] - newh)/2)
	#imgc = img.copy()*(2.0/255.0) - 1.0
	imgc = img.copy()*pmul + padd

	imgb[offy:offy+newh,offx:offx+neww,:] = resize(imgc,(newh,neww),1.0)
	#im = imgb[:,:,(2,1,0)]
	return imgb

def runInference(img, api, hdl):
	imgw = img.shape[1]
	imgh = img.shape[0]
	out_buf = bytearray(1000*4)
	out = np.frombuffer(out_buf, dtype=numpy.float32)
	#convert image to tensor format (RGB in seperate planes)
	img_r = img[:,:,0]
	img_g = img[:,:,1]
	img_b = img[:,:,2]
	img_t = np.concatenate((img_r, img_g, img_b), 0)

	status = api.annCopyToInferenceInput(hdl, np.ascontiguousarray(img_t, dtype=np.float32), (img.shape[0]*img.shape[1]*3*4), 0)
	#print('INFO: annCopyToInferenceInput status %d'  %(status))
	status = api.annRunInference(hdl, 1)
	#print('INFO: annRunInference status %d ' %(status))
	status = api.annCopyFromInferenceOutput(hdl, np.ascontiguousarray(out, dtype=np.float32), len(out_buf))
	#print('INFO: annCopyFromInferenceOutput status %d' %(status))
	return out

def predict_fn(images):
	results = np.zeros(shape=(len(images), 1000))
	for i in range(len(images)):
		results[i] = runInference(images[i])    
	return results

class App(QWidget):
 
	def __init__(self):
		super(App, self).__init__()
		self.initUI()

	def initUI(self):

		self.title = 'MIVision IMAGE CLASSIFICATION'
		self.setWindowTitle(self.title)
		cwd = os.getcwd() 
		self.setWindowIcon(QIcon(cwd + '/icons/amd-logo-150x150.jpg'))
				
		#pixmap = QPixmap(cwd + '/detectionExample/icons/amd-logo.jpg')

		# Create widgets
		self.label  = QLabel('Most Recent Images Found', self)      
		self.label.setStyleSheet("font: bold 14pt TIMES NEW ROMAN")
		self.label.setWordWrap(True)
		self.label.setAlignment(Qt.AlignCenter) 
		#self.imageWidget = QLabel()
		#self.imageWidget.setPixmap(pixmap)
		self.tableWidget = QTableWidget()

		self.layout = QGridLayout()
		self.layout.setSpacing(5)
		#self.layout.addWidget(self.imageWidget,1,0)
		self.layout.addWidget(self.label,2,0)
		self.layout.addWidget(self.tableWidget,3,0) 

		self.setLayout(self.layout) 
		
		self.setGeometry(0, 0, 1200, 300)

	def createTable(self,data):
		self.show()

		self.tableWidget.setRowCount(0)
		self.tableWidget.setColumnCount(10)
		self.tableWidget.setHorizontalHeaderLabels(['Occurence','Top 1', 'Confidence', '','Top 2', 'Confidence', '','Top 3', 'Confidence',''])
		for row_number in xrange(0,min(5,len(data))): 
			name_1 = str(data[row_number][1]).split(" ",1)
			name_2 = str(data[row_number][4]).split(" ",1)
			name_3 = str(data[row_number][7]).split(" ",1)           
			conf_1 = str(int(round(float(data[row_number][2]), 2)*100)) + "%"
			conf_2 = str(int(round(float(data[row_number][5]), 2)*100)) + "%"
			conf_3 = str(int(round(float(data[row_number][8]), 2)*100)) + "%"
			
			row_number =  self.tableWidget.rowCount()
			self.tableWidget.insertRow(row_number)
			self.tableWidget.setItem(row_number, 1, QTableWidgetItem(name_1[1]))
			#self.tableWidget.item(row_number,1).setBackground(QColor('#388E8E'))
			self.tableWidget.setItem(row_number, 2, QTableWidgetItem(conf_1))
			self.tableWidget.setItem(row_number, 4, QTableWidgetItem(name_2[1]))
			self.tableWidget.setItem(row_number, 5, QTableWidgetItem(conf_2)) 
			self.tableWidget.setItem(row_number, 7, QTableWidgetItem(name_3[1]))
			self.tableWidget.setItem(row_number, 8, QTableWidgetItem(conf_3))
		self.tableWidget.setAlternatingRowColors(True)
		self.tableWidget.setItem(0,0,QTableWidgetItem("Most Recent"))
		self.tableWidget.setItem(2,0,QTableWidgetItem("Older"))
		self.tableWidget.setItem(4,0,QTableWidgetItem("Oldest"))		
		self.tableWidget.resizeColumnsToContents()
		self.tableWidget.setColumnWidth(3,3)
		self.tableWidget.setColumnWidth(6,3)
		self.tableWidget.setColumnWidth(9,3)
		
		
		# Show widget
		self.show()

def show_legend():
	keys = ['key','1', '2', 'f', 'q' ,'Space Bar', 'c']
	modes = ['mode','folder 1 - one iteration' , 'folder 2 - one iteration' , 'folder - always', 'quit','pause/play', 'camera mode']
	fontScale = 1
	thickness = 1
	legendGeometry = (300,500)
	legend = np.zeros(legendGeometry, dtype=np.uint8)
	
	for i in xrange(len(keys)):
		size = cv2.getTextSize(keys[i], cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)
		width = size[0][0]
		height = size[0][1]

		#cv2.rectangle(legend, (5, (i * 25) + 17),(300, (i * 25) + 25),(160,82,45),-1)	
		cv2.putText(legend, keys[i], (5, ((i+2)*25)), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (255,255,255), thickness,2 )
		cv2.putText(legend, modes[i], (150, (i+2) * 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (255,255,255), thickness,2 )
		#cv2.rectangle(legend, (5, (i * 25) + 17),(300, (i * 25) + 25),(0,0,255),-1)
	
	cv2.imshow("MIVision Classification Legend", legend) 	 	

def image_function():
	imagefile = args.image
	# image preprocess
	top_indeces = []
	top_labels = []
	img = cv2.imread(imagefile)
	imgb = PreprocessImage(img, inp_dim, pmul, padd)
	start = datetime.now()
	output = runInference(imgb, api, hdl)
	for x in output.argsort()[-3:]:
		print (x, names[x], output[x])
		top_indeces.append(x)
		top_labels.append(names[x])

	txt =  top_labels[2]   
	size = cv2.getTextSize(top_labels[2], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, 2)
	t_width = size[0][0]
	t_height = size[0][1]
	cv2.rectangle(img, (50, 50), (t_width+50,t_height+50), (192,192,128), -1)
	cv2.putText(img,txt,(2,52),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.1,(20,20,20),2)
	cv2.imshow('MIVision Classification Live', img)
	key = cv2.waitKey()
	#cv2.destroyAllWindows()
	#AnnInferenceLib.annReleaseContext(ctypes.c_void_p(hdl))
	api.annReleaseInference(hdl)
	exit()

def imagefolder_function(imagedir, outputdir):
	show_legend()
	count = 0  
	# image preprocess
	start = datetime.now()
	dictnames = ['input_image_name', 'class_top1', 'class_name_top1', 'confidence_top1', 'class_top2', 'class_name_top2', 'confidence_top2', 'class_top3', 'class_name_top3', 'confidence_top3']
	csvFile = open('classification_out.csv', 'w')
	with csvFile:
		writer = csv.DictWriter(csvFile, fieldnames=dictnames)
		writer.writeheader()
		for image in sorted(os.listdir(imagedir)):
				
			top_indeces = []
			top_labels = []
			top_prob = []
			print('Processing Image ' + image)
			img = cv2.imread(imagedir + image)
			imgb = PreprocessImage(img, inp_dim, pmul, padd)
			output = runInference(imgb, api, hdl)
			for x in output.argsort()[-3:]:
				top_indeces.append(x)
				top_labels.append(names[x])
				top_prob.append(output[x])
			writer.writerow({'input_image_name': image, 'class_top1':top_indeces[2], 'class_name_top1':top_labels[2], 'confidence_top1': top_prob[2],\
							'class_top2':top_indeces[1], 'class_name_top2':top_labels[1], 'confidence_top2': top_prob[1],\
							'class_top3':top_indeces[0], 'class_name_top3':top_labels[0], 'confidence_top3': top_prob[0],})
			txt1 =  top_labels[2].split(' ',1)
			txt = txt1[1]
			#print txt 
			size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, 2)
			t_width = size[0][0]
			t_height = size[0][1]
			cv2.rectangle(img, (5, 20), (t_width+50,t_height+50), (192,192,128), -1)
			cv2.putText(img,txt,(2,52),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(20,20,20),2)

			path = os.path.join(outputdir ,  'classification-output_'+ str(count) + '.jpg')
			cv2.imshow('MIVision Classification Live', img)
			time.sleep(0.8)
			#cv2.waitKey(1000)
			cv2.imwrite(path,img)
			count += 1
			key = cv2.waitKey(1)
			if key & 0xFF == ord('q'):
				api.annReleaseInference(hdl)
				exit()
			if key & 0xFF == ord('c'):
				#cap = cv2.VideoCapture(0)
				camera_function(0)
			if key == 32:
				if cv2.waitKey(0) == 32:
					continue
			
		#cv2.destroyAllWindows()
		#AnnInferenceLib.annReleaseContext(ctypes.c_void_p(hdl))
		end = datetime.now()
		elapsedTime = end-start
		print ('total time is " milliseconds', elapsedTime.total_seconds()*1000)


def camera_function(capmode):

	show_legend()
	print ('Capturing Live')
	cap = cv2.VideoCapture(0)
	assert cap.isOpened(), 'Cannot capture source'    
	frames = 0
	start = time.time()
	data = []
	dictnames = ['class_top1', 'class_name_top1', 'confidence_top1', 'class_top2', 'class_name_top2', 'confidence_top2', 'class_top3', 'class_name_top3', 'confidence_top3']
	with open('history_file.csv', 'w+') as csvHistoryFile:
		writer = csv.DictWriter(csvHistoryFile, fieldnames=dictnames)
		writer.writeheader()
		while cap.isOpened():        
			top_indeces = []
			top_labels = []
			top_prob = []
			top_hei = []
			ret, frame = cap.read()
			if ret:
				frame = cv2.flip(frame, 1)                
				imgb = PreprocessImage(frame, inp_dim, pmul, padd)
				output = runInference(imgb, api, hdl)
				for x in output.argsort()[-3:]:
					top_indeces.append(x)
					top_labels.append(names[x])
					top_prob.append(output[x])
					top_hei.append(hierarchies[x])
				#print ("outside if, capmode = " , capmode)
				if int(capmode) == 0:
					#draw a rectangle on the image at top    
					txt =  top_labels[2].lstrip(top_labels[2].split(' ')[0])   
					size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
					t_width = size[0][0]
					t_height = size[0][1]
					cv2.rectangle(frame, (10, 10), (t_width+2,t_height+16), (192,192,128), -1)
					cv2.putText(frame,txt,(10,t_height+10),cv2.FONT_HERSHEY_DUPLEX,0.7,(20,20,20),2)
					cv2.imshow('MIVision Classification Live', frame)
				
				elif int(capmode) == 1:
					#print ("in capmode 1")
					txt =  top_labels[2].lstrip(top_labels[2].split(' ')[0])   
					txt1 =  top_hei[2].replace(',', ' ')
					size = cv2.getTextSize(top_hei[2], cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
					t_width = size[0][0]
					t_height = size[0][1]
					t_height1 = size[0][1]*2
					cv2.rectangle(frame, (10, 10), (t_width+10,t_height1+16), (192,192,128), -1)
					cv2.putText(frame,txt,(10,t_height+10),cv2.FONT_HERSHEY_DUPLEX,0.7,(20,20,20),2)
					cv2.putText(frame,txt1,(10,t_height1+10),cv2.FONT_HERSHEY_DUPLEX,0.7,(20,20,20),2)
					cv2.imshow('MIVision Classification Live', frame)
				
				writer.writerow({'class_top1':top_indeces[2], 'class_name_top1':top_labels[2], 'confidence_top1': top_prob[2],\
							'class_top2':top_indeces[1], 'class_name_top2':top_labels[1], 'confidence_top2': top_prob[1],\
							'class_top3':top_indeces[0], 'class_name_top3':top_labels[0], 'confidence_top3': top_prob[0],})
				
				frames += 1
				if (frames %16 == 0):
					print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
					csvHistoryFile.seek(0)
					resultCSV = csv.reader(csvHistoryFile)
					next(resultCSV,None) # skip header
					resultDataBase = [r for r in resultCSV]
					numElements = len(resultDataBase)

					resultDataBase = [row for row in resultDataBase if row and len(row) == 9]
					#for row in resultDataBase:
					#    print row
					if numElements > 0: 
						
						for i in xrange(len(resultDataBase) - 1, 0 , -1):
							if resultDataBase[i][0] == resultDataBase[i-1][0]: 
								del resultDataBase[i]
						
						data.insert(0,resultDataBase[0])
						for i in xrange(len(data) - 1, 0 , -1):
							if data[i][0] == data[i-1][0]: 
								del data[i]
						
						qt_tryme.createTable(data)
						#time.sleep(0.2)
						csvHistoryFile.seek(0)
						writer = csv.DictWriter(csvHistoryFile, fieldnames=dictnames)
						writer.writeheader()


				key = cv2.waitKey(1)
				if key & 0xFF == ord('q'):
					api.annReleaseInference(hdl)
					cap.release()
					exit()

				if key == 32:
					if cv2.waitKey(0) == 32:
						continue

				if key & 0xFF == ord('1'):
					cap.release()
					imagedir  = os.getcwd() + '/images/'
					outputdir = os.getcwd() + '/outputFolder_1/'
					if not os.path.exists(outputdir):
						os.makedirs(outputdir)
					imagefolder_function(imagedir, outputdir)
			   		cap = cv2.VideoCapture(0)

				if key & 0xFF == ord('2'):
					cap.release()
					imagedir  = os.getcwd() + '/images_2/'
					outputdir = os.getcwd() + '/outputFolder_2/'
					if not os.path.exists(outputdir):
						os.makedirs(outputdir)
					imagefolder_function(imagedir, outputdir)
					cap = cv2.VideoCapture(0)

				if key & 0xFF == ord('f'):
					cap.release()
					imagedir  = os.getcwd() + '/images/'
					outputdir = os.getcwd() + '/outputFolder/'
					if not os.path.exists(outputdir):
						os.makedirs(outputdir)
					flag = True
					while (flag):
						imagefolder_function(imagedir, outputdir)
				
			else:
				break

		api.annReleaseInference(hdl)
		exit()





if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image', dest='image', type=str,
						default='./data/.jpg', help='An image path')
	parser.add_argument('--capture', dest='capmode', type=str,
						default=0, help='CaptureMode')
	parser.add_argument('--imagefolder', dest='folder', type=str,
						default='./images/' + 'images', help='A directory with images')
	parser.add_argument('--annpythonlib', dest='pyhtonlib', type=str,
						default='./libannpython.so', help='pythonlib')
	parser.add_argument('--weights', dest='weightsfile', type=str,
						default='./weights.bin', help='A directory with images.')
	parser.add_argument('--labels', dest='labelfile', type=str,
						default='./labels.txt', help='file with labels')
	parser.add_argument('--hierarchy', dest='hierarchyfile', type=str,
						default='./hier-may14.csv', help='file with labels')
	parser.add_argument('--resultsfolder', dest='resultfolder', type=str,
						default=os.getcwd() + '/outputFolder', help='A directory with classified images.')
	parser.add_argument('--pm', dest='pm', type=str,
						help='preprocessMul')
	parser.add_argument('--pa', dest='pa', type=str,
						help='preprocessAdd')
	args = parser.parse_args()

	
	
	
	synsetfile = args.labelfile
	weightsfile = args.weightsfile
	annpythonlib = args.pyhtonlib
	hierarchyfile = args.hierarchyfile
	pmul = float(args.pm)
	padd = int(args.pa)
	
	app = QApplication(sys.argv)
	qt_tryme = App() 

	api = AnnAPI(annpythonlib)
	input_info,output_info = api.annQueryInference().decode("utf-8").split(';')
	input,name,ni,ci,hi,wi = input_info.split(',')
	hdl = api.annCreateInference(weightsfile)
	inp_dim = (int(wi), int(hi))

	#read synset names
	if synsetfile:
		fp = open(synsetfile, 'r')
		names = fp.readlines()
		names = [x.strip() for x in names]
		fp.close()

	#read hierarchy names
	if hierarchyfile:
		fp = open(hierarchyfile, 'r')
		hierarchies = fp.readlines()
		hierarchies = [x.strip() for x in hierarchies]
		fp.close()

	if sys.argv[1] == '--image':        
		image_function()
		

	elif sys.argv[1] == '--imagefolder':  
		outputdir = args.resultfolder
		if not os.path.exists(outputdir):
			os.makedirs(outputdir);      
		imagefolder = args.folder
		imagefolder_function(imagefolder, outputdir)
		

	elif sys.argv[1] == '--capture':
		capmode = args.capmode
		camera_function(capmode)

	#api.annReleaseInference(hdl)
	exit()
