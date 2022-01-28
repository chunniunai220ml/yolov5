import torch
import tensorrt as trt


def pt2onnx(model,im,file,opset):
    torch.onnx.export(model, im, file, verbose=True, opset_version=opset,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch'},  # , 2: 'height', 3: 'width',shape(1,3,640,640)
                                        'output': {0: 'batch'}  # , 1: 'anchors',shape(1,25200,85)
                                        })


def export_engine(file,half,workspace=4, verbose=False, prefix='TensorRT:'):
    try:
        import tensorrt as trt
        onnx = file.with_suffix('.onnx')
        assert onnx.exists(), f'failed to export ONNX file: {onnx}'
        print(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
        f = file.with_suffix('.engine')  # TensorRT engine file
        loggers = trt.Logger(trt.Logger.INFO)
        if verbose:
            loggers.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger=loggers)
        
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        # dt=trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION=trt.float32
        network =builder.create_network(flag)

        parser = trt.OnnxParser(network, logger=loggers)
        sucess=parser.parse_from_file(str(onnx))
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        if not sucess:
            raise RuntimeError(f'failed to load ONNX file: {onnx}')
        #set input shape to dynamic batchsize
        input = network.get_input(0)
        input.shape = (-1, 3, 640, 640)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 31
        inputs =[network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        # print(outputs)
        print(f'{prefix} Network Description:')
        for inp in inputs:
            print(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            print(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')
        print(f'{prefix} building FP{16 if half else 32} engine in {f}')
        
        if half:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
            # inspector = engine.create_engine_inspector()
            # print(inspector.get_engine_information(LayerInformationFormat.JSON))  
            t.write(engine.serialize())
        print(f'{prefix} export success, saved as {f}') #({file_size(f):.1f} MB)')

    except Exception as e:
        print(f'\n{prefix} export failure: {e}')


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


def torch_device_to_trt(device):
    if device.type == torch.device('cuda').type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device('cpu').type:
        return trt.TensorLocation.HOST
    else:
        return TypeError('%s is not supported by tensorrt' % device)

def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)       


class TRTModel(object):
    
    def __init__(self, engine_path, input_names=None, output_names=None, final_shapes=None):

        # load engine
        self.logger = trt.Logger()
        self.runtime = trt.Runtime(self.logger)
        try:
            with open(engine_path, 'rb') as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except BaseException as e:
            import traceback
            traceback.print_exc()
            import os
            os._exit(1)

        self.context = self.engine.create_execution_context()        
        if input_names is None:
            self.input_names = self._trt_input_names()
        else:
            self.input_names = input_names

        if output_names is None:
            self.output_names = self._trt_output_names()
        else:
            self.output_names = output_names

        self.final_shapes = final_shapes
        # torch.cuda.device('cuda').make_context()
        self.stream = torch.cuda.current_stream(device='cuda')
        def _input_binding_indices(self):
            return [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]

    def _output_binding_indices(self):
        return [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]

    def _trt_input_names(self):
        return [self.engine.get_binding_name(i) for i in self._input_binding_indices()]

    def _trt_output_names(self):
        return [self.engine.get_binding_name(i) for i in self._output_binding_indices()]

    def create_output_buffers(self, batch_size):
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            if self.final_shapes is not None:
                shape = (batch_size,) + self.final_shapes[i]
            else:
                assert self.context.get_binding_shape(idx)[0]!=-1 and self.context.get_binding_shape(idx)[0]!=0
                shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
        return outputs
    def execute(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        # map input bindings
        inputs_torch = [None] * len(self.input_names)
        for i, name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(name)
            self.context.set_binding_shape(idx, inputs[0].shape)
            # convert to appropriate format
            inputs_torch[i] = inputs[i]
            inputs_torch[i] = inputs_torch[i].to(torch_device_from_trt(self.engine.get_location(idx)))
            inputs_torch[i] = inputs_torch[i].type(torch_dtype_from_trt(self.engine.get_binding_dtype(idx)))
            bindings[idx] = int(inputs_torch[i].data_ptr())

        output_buffers = self.create_output_buffers(batch_size)
        # map output bindings
        for i, name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(name)
            if idx>len(bindings):
                bindings[-1] = int(output_buffers[i].data_ptr())
            else:
                bindings[idx] = int(output_buffers[i].data_ptr())
        outputs = self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.cuda_stream)
        return outputs


def inferdemo(imgfp,trt_path='runs/train/exp5/weights/best.engine',bs=4,half=True):
    import cv2
    import time,os
    import numpy as np
    from utils.general import non_max_suppression
    from utils.plots import Annotator
    from utils.general import scale_coords
    from utils.datasets import LoadImages
    from pathlib import Path
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    trt_path='runs/train/exp5/weights/best.engine' 
    trtmodel = TRTModel(trt_path,input_names=["images"],output_names=["350","416","482","output"],final_shapes=[ (3, 80, 80, 6), ( 3, 40, 40, 6), (3, 20, 20, 6),(25200, 6)])
    # img = cv2.resize(imgfp, imgsz)
    device='cuda' 
    dataset = LoadImages(imgfp, stride=64, auto=False)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() 
        img /= 255.0
        print('img.shape: ',img.shape)
        if len(img.shape) == 3:
            c,w,h=img.shape
            img=img.expand((bs,c,w,h))
            im0s=np.array([im0s]* bs)
        st1=time.time()
        trt_pred = trtmodel(img)[-1] #trtmodel return [pred],so take[0]
        et1=time.time()-st1
        print('{0} images infer time: '.format(bs),et1) 
        st2=time.time()
        preds=non_max_suppression(trt_pred, conf_thres=0.25, iou_thres=0.45)  #[non_max_suppression(tp, conf_thres=0.25, iou_thres=0.45) for tp in trt_preds]
        et2=time.time()-st2
        print('nms time: ',et2)
        for i, preddet in enumerate(preds):
            p, s, im0, frame = path, '', im0s[i].copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = 'trtimbs/'+str(i)+'-'+str(p.name)  # img.jpg
            annotator = Annotator(im0, line_width=3, example='person')
            if len(preddet):
                det = scale_coords(img.shape[2:], preddet[:, :], im0.shape)
                det[:,:4]=det[:,:4].round()
                # print(det)
                for *xyxy, conf, cls in reversed(det):
                    annotator.box_label(xyxy, f'{conf:.2f}', color=(255,0,0))
            
            cv2.imwrite(save_path, im0)       



    
            
        