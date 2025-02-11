#----------------------------------------------------------------------------------------------------------
#  class ONNXmodel 
#
#  loads and runs  ONNX models
#
#----------------------------------------------------------------------------------------------------------

import sys,os
import numpy as np
import time
from tqdm import tqdm


import onnxruntime   # pip install onnxruntime-gpu


# cuda sudo apt install nvidia-cuda-toolkit
# check cuda version in terminal with $ nvidia-smi 
# CUDA 12.0
import pycuda.driver as cuda
import torch

try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

print ("")
print ("VERSIONS")
print("tensorrt:",trt.__version__) 
print("pycuda:",pycuda.VERSION_TEXT)
print("cuda version:",cuda.get_version())
print("cuda driver version:",cuda.get_driver_version())
print ("")

class ONNXmodel:
    def __init__(self,trt_file_path):
        self.name="ONNX"
        self.providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(ONNX_FILE_PATH, None,providers=self.providers)
         # get the name of the first input of the model
        self.input_name = self.session.get_inputs()[0].name  
        print('Input Name:', self.input_name)   
 
    def inference(self,x):
        output =self.session.run([], {self.input_name:x})[0] 
        output=output[0,:,:,:]
        return output

    def test(self,x,target):
        y=self.inference(x)
        return y



    def rmse(self,target,pred,borderpercent=6.25):
        _,sx,sy=pred.shape
        bx=int(sx*borderpercent/100)
        by=int(sy*borderpercent/100)
        target=target[:,bx:-bx,by:-by]
        pred=pred[:,bx:-bx,by:-by]
        t=target.reshape(-1)
        p=pred.reshape(-1)
        rmse1= np.sqrt(np.mean((p-t)**2))
        return rmse1

    def empty(self):
         with torch.no_grad():
            torch.cuda.empty_cache()



def empty():
    torch.cuda.empty_cache()


def process_dataset(model, dataset,writer=None, maxlen=10):
    print("processing images")

    #warmup
    for i in range(5):
        image_pair, vects,target= dataset[i]
        pred = model.inference(image_pair)
    
    # Length
    N = len(dataset)
    if N>maxlen: N=maxlen
    grid=dataset.grid

    # Process
    print("process ",N,'images')
    t0=time.time()
    er0m=0;er0=0
    erlm=0;erl=0
    erdm=0;erd=0
    for i in tqdm(range(N)):
        image_pair, vects,target= dataset[i]
        pred = model.inference(image_pair)
        res=pred[:,grid[0],grid[1]]
        if target is not None:
            trg=target[:,grid[0],grid[1]]
            er0=model.rmse(target,pred)
            erl=model.rmse(trg,res)
            if vects is not None:
                erd=model.rmse(trg,vects)
        er0m=er0m+er0
        erlm=erlm+erl
        erdm=erdm+erd
        if writer is not None: 
            writer.write(pred)
    dt=time.time()-t0
    print("time used:",round(dt/N*1000,2),'[ms] per pair   mean er0:',round(er0m/N,3),'  erl:',round(erlm/N,3), ' erd:',round(erdm/N,3),'pixels') 
    print("")

    return 


if __name__ == "__main__" :
    import  dataset as dt
    from plot import plot_res
    
    print("DATA")
    model_file='TRT_models/2022/Lima_L6_dyn.onnx'
    image_path='data/davis_ckb/PIV_n10_s384_maxd10_ckb_v1'
    vector_path='data/davis_ckb/PIV_n10_s384_maxd10_ckb_v1/PIV_MPd(2x8x8_50%ov)'
    print("model:",model_file,'  exist:',os.path.exists(model_file))
    print("images:",image_path,'  exist:',os.path.exists(image_path))
    print("vectors:",vector_path,'  exist:',os.path.exists(vector_path))
    dataI=dt.LaVisionIMx(image_path)
    dataV=dt.LaVisionVC7(vector_path)
    ims=dataI[0]
    vects,grid=dataV[0]
    print('ims:',ims.shape)
    print('vects:',vects.shape)    
    print('grid:',grid.shape)
    print('')
    
    print('LOAD MODEL')
    torch.cuda.empty_cache()
    shape=dataI.shape
    model=ONNXmodel(model_file)
    print('')

    print('INFERENCE')
    inr=0
    ims=dataI[inr]
    vects=dataV[inr]
    grid=dataV.grid
    pred=model.inference(ims)
    res=pred[:,grid[0],grid[1]]
    er=model.rmse(vects,res)
    print("image:",inr,'  rmse:',er)
    del model

    
    print('PROCESS Dataset')
    size=256
    dataset_path='data/davis_rnd/PIV_n10_s{}_maxd10_rnd_v1'.format(size)
    result_path=dataset_path+'/lima'
    dataset=dt.LaVisionDataset(dataset_path) 
    shape=dataset.shapeI
    model=TRTmodel(model_file,dynamic=True,shape=shape)
    N=len(dataset)
    shape=dataset.shapeV
    writer=dt.H5writer('results/davis_rnd/PIV_n10_s256_maxd10_rnd_v1.h5',N,shape)
    process_dataset(model,dataset,writer)
      
    
    del model  #  17483MiB => 1449MiB
    empty()    #   1449MiB => 1455MiB
    print('')  
    # exit     #   1455MiB =>   41MiB




