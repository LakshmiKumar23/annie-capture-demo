import sys,os,time,csv,getopt,cv2,argparse
import numpy, ctypes, array
import numpy as np
from datetime import datetime
from ctypes import cdll, c_char_p
from skimage.transform import resize
from numpy.ctypeslib import ndpointer
import ntpath
import scipy.misc
from PIL import Image


AnnInferenceLib = ctypes.cdll.LoadLibrary('/home/rajy/work/inceptionv4/build/libannmodule.so')
inf_fun = AnnInferenceLib.annRunInference
inf_fun.restype = ctypes.c_int
inf_fun.argtypes = [ctypes.c_void_p,
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_size_t]
hdl = 0

def PreprocessImage(img, dim):
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
    imgc = img.copy()*(2.0/255.0) - 1.0

    #print('INFO:: imw: %d imh: %d dim0: %d dim1:%d newW:%d newH:%d offx:%d offy: %d' % (imgw, imgh, dim[0], dim[1], neww, newh, offx, offy))
    imgb[offy:offy+newh,offx:offx+neww,:] = resize(imgc,(newh,neww),1.0)
    #im = imgb[:,:,(2,1,0)]
    return imgb

def runInference(img):
    global hdl
    imgw = img.shape[1]
    imgh = img.shape[0]
    #proc_images.append(im)
    out_buf = bytearray(1000*4)
    #out_buf = memoryview(out_buf)
    out = np.frombuffer(out_buf, dtype=numpy.float32)
    #im = im.astype(np.float32)
    inf_fun(hdl, np.ascontiguousarray(img, dtype=np.float32), (img.shape[0]*img.shape[1]*3*4), np.ascontiguousarray(out, dtype=np.float32), len(out_buf))
    return out

def predict_fn(images):
    results = np.zeros(shape=(len(images), 1000))
    for i in range(len(images)):
        results[i] = runInference(images[i])    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='image', type=str,
                        default='./images/dog.jpg', help='An image path.')
    parser.add_argument('--video', dest='video', type=str,
                        default='./videos/car.avi', help='A video path.')
    parser.add_argument('--imagefolder', dest='imagefolder', type=str,
                        default='./', help='A directory with images.')
    parser.add_argument('--resultsfolder', dest='resultfolder', type=str,
                        default='./', help='A directory with images.')
    parser.add_argument('--labels', dest='labelfile', type=str,
                        default='./labels.txt', help='file with labels')
    args = parser.parse_args()

    imagefile = args.image
    videofile = args.video
    imagedir  = args.imagefolder
    outputdir = args.resultfolder
    synsetfile = args.labelfile
    images = []
    proc_images = []
    AnnInferenceLib.annCreateContext.argtype = [ctypes.c_char_p]
    data_folder = "/home/rajy/work/inceptionv4"
    b_data_folder = data_folder.encode('utf-8')
    global hdl
    hdl = AnnInferenceLib.annCreateContext(b_data_folder)
    inp_dim = (299, 299)
    #read synset names
    if synsetfile:
        fp = open(synsetfile, 'r')
        names = fp.readlines()
        names = [x.strip() for x in names]
        fp.close()

    if sys.argv[1] == '--image':
        # image preprocess
        top_indeces = []
        top_labels = []
        img = cv2.imread(imagefile)
        imgb = PreprocessImage(img, inp_dim)
        start = datetime.now()
        output = runInference(imgb)
        f = open('out_file.f32', 'wb')
        f.write(output.tobytes())
        f.close()        
        for x in output.argsort()[-3:]:
            print (x, names[x], output[x])
            top_indeces.append(x)
            top_labels.append(names[x])

        txt =  top_labels[2]   
        size = cv2.getTextSize(top_labels[2], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, 2)
        t_width = size[0][0]
        t_height = size[0][1]
        cv2.rectangle(img, (50, 50), (t_width+50,t_height+50), (192,192,128), -1)
        cv2.putText(img,txt,(52,52),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.1,(20,20,20),2)
        cv2.imshow('Demo', img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            AnnInferenceLib.annReleaseContext(ctypes.c_void_p(hdl))
            exit()
        #cv2.destroyAllWindows()
        AnnInferenceLib.annReleaseContext(ctypes.c_void_p(hdl))
        exit()

    elif sys.argv[1] == '--video':
        count = 0
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), 'Cannot capture source'    
        frames = 0
        start = time.time()    
        while cap.isOpened():        
            top_indeces = []
            top_labels = []
            ret, frame = cap.read()
            if ret:
                imgb = PreprocessImage(frame, inp_dim)
                output = runInference(imgb)
                for x in output.argsort()[-3:]:
                    print (x, names[x], output[x])
                    top_indeces.append(x)
                    top_labels.append(names[x])
                #draw a rectangle on the image at top    
                txt =  top_labels[2].lstrip(top_labels[2].split(' ')[0])   
                size = cv2.getTextSize(top_labels[2], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, 2)
                t_width = size[0][0]
                t_height = size[0][1]
                cv2.rectangle(frame, (10, 10), (t_width+10,t_height+10), (192,192,128), -1)
                cv2.putText(frame,txt,(10,t_height+5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.1,(20,20,20),2)
                cv2.imshow('AMD InceptionV4 Live', frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            else:
                break

        AnnInferenceLib.annReleaseContext(ctypes.c_void_p(hdl))
        exit()

