 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import warnings
from comfy_api.latest import  io
import folder_paths

from .node_utils import  tensor2pillist,clear_comfyui_cache
from .SenseNova.examples.editing.inference import load_sensenova_model,infer_sensenova_edit
from .SenseNova.examples.t2i.inference import infer_sensenova_t2i,SUPPORTED_RESOLUTIONS
from .SenseNova.examples.interleave.inference import infer_sensenova_interleave,SUPPORTED_RESOLUTIONS as SUPPORTED_RESOLUTIONS_interleave
from .SenseNova.examples.vqa.inference import infer_sensenova_vqa


# 设备检测：支持 CUDA、XPU、MPS，如果只有 CPU 则发出醒目警告
def _detect_device():
    """检测并返回最佳可用设备，支持 CUDA、XPU、MPS"""
    # 优先检测 CUDA
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    
    # 检测 XPU (Intel GPU)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu:0")
    
    # 检测 MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    
    # 只有 CPU 可用，发出醒目警告
    warning_msg = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                          ⚠️  WARNING: CPU MODE  ⚠️                           ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  No GPU device detected! The model will run on CPU, which is EXTREMELY slow. ║
    ║                                                                              ║
    ║  Supported GPU devices:                                                      ║
    ║    - NVIDIA CUDA (torch.cuda.is_available())                                ║
    ║    - Intel XPU   (torch.xpu.is_available())                                 ║
    ║    - Apple MPS   (torch.backends.mps.is_available())                        ║
    ║                                                                              ║
    ║  For optimal performance, please ensure:                                    ║
    ║    1. GPU drivers are properly installed                                    ║
    ║    2. PyTorch is installed with GPU support                                 ║
    ║       - CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121 ║
    ║       - XPU:  pip install torch --index-url https://pytorch-extension.intel.com/ipex-whl-stable-xpu ║
    ║                                                                              ║
    ║  Running on CPU may take HOURS instead of SECONDS for image generation!     ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    warnings.warn(warning_msg, UserWarning, stacklevel=2)
    print(warning_msg)  # 确保警告一定会显示
    return torch.device("cpu")

device = _detect_device()

MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))
weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir

class SenseNova_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNova_SM_Model",
            display_name="SenseNova_SM_Model",
            category="SenseNova",
            inputs=[
                io.Combo.Input("diffusion_models",options= ["none"] + folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),
                io.Combo.Input("lora",options= ["none"] + folder_paths.get_filename_list("loras")),
                io.Combo.Input("attn_backend",options= ["auto", "flash", "sdpa"]),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, diffusion_models,gguf,lora,attn_backend) -> io.NodeOutput:
        clear_comfyui_cache()
        dit_path=folder_paths.get_full_path("diffusion_models",diffusion_models) if diffusion_models != "none" else None
        gguf_path=folder_paths.get_full_path("gguf",gguf) if gguf != "none" else None
        lora_path=folder_paths.get_full_path("loras",lora) if lora != "none" else None
        model_path=dit_path or gguf_path
        model=load_sensenova_model(model_path,device,node_cr_path,attn_backend,lora_path=lora_path)
        return io.NodeOutput(model)
    

class SenseNova_SM_Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNova_SM_Sampler",
            display_name="SenseNova_SM_Sampler",
            category="SenseNova",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("img_mode",options= ["edit",  "interleave", "vqa"]),
                io.String.Input("prompt",default="a photo of a cat",multiline=True),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Int.Input("steps", default=8, min=1, max=10000, step=1),
                io.Combo.Input("target_pixels",options= ["1:1", "16:9", "9:16", "3:2","2:3","4:3","3:4","1:2","2:1","1:3","3:1"]),
                io.Float.Input("cfg", default=1.0, min=0.0, max=10.0, step=0.1, round=0.01,),
                io.Float.Input("img_cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,),
                io.Float.Input("timestep_shift", default=3.0, min=-1.0, max=10.0, step=0.1, ),
                io.Int.Input("batch_size", default=1, min=1, max=64,step=1),
                io.Int.Input("prefetch_count", default=1, min=0, max=64,step=1),
                io.Int.Input("interleave_max", default=4, min=1, max=MAX_SEED),
                io.Combo.Input("cfg_norm",options= ["none", "global", "channel"]),
                io.Boolean.Input("enhance", default=False),
                io.Boolean.Input("think_mode", default=False),
                io.Boolean.Input("do_sample", default=True),
                io.Int.Input("max_new_tokens", default=1024, min=256, max=10241024,step=1),
                io.Float.Input("temperature", default=0.7, min=0.0, max=1.0, step=0.1,),
                io.Float.Input("top_p", default=0.9, min=0.0, max=1.0, step=0.1,),
                io.Int.Input("top_k", default=0, min=0, max=1024,step=1),
                io.Float.Input("repetition_penalty", default=0.0, min=0.0, max=10.0, step=0.1,),
                io.Image.Input("image",optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="text"),
            ],
        )
    
    @classmethod
    def execute(cls, model, img_mode,prompt,seed, steps, target_pixels,cfg,img_cfg,timestep_shift,batch_size,prefetch_count,interleave_max,cfg_norm,enhance,
                think_mode,do_sample,max_new_tokens,temperature,top_p,top_k,repetition_penalty,image=None) -> io.NodeOutput:
        clear_comfyui_cache()
        top_k=None if top_k==0 else top_k
        repetition_penalty=repetition_penalty if repetition_penalty>0.0 else None
        cfg_interval=[0.0,1.0]
        width,height=SUPPORTED_RESOLUTIONS_interleave[target_pixels] if img_mode=="interleave" else SUPPORTED_RESOLUTIONS[target_pixels]
        images=tensor2pillist(image) if image is not None else None
        
        if prefetch_count==0:
            model.model.to(device)
            prefetch_count=None
        
        if images is not None:
            print(f"infer_mode is : {img_mode}")
            if "edit"==img_mode:
                image,text=infer_sensenova_edit(model,prompt,cfg,cfg_norm,steps,batch_size,timestep_shift,img_cfg,cfg_interval,width,height,images,target_pixels,seed,prefetch_count,think_mode)
            elif "vqa"==img_mode:
                image=torch.zeros((1,height, width,3))
                text=infer_sensenova_vqa(model,prompt,images[0],max_new_tokens,do_sample,temperature,top_p,top_k,repetition_penalty,prefetch_count)
            else:
                text,image=infer_sensenova_interleave(model,prompt,cfg,steps,timestep_shift,img_cfg,images,cfg_interval,width,height,think_mode,seed,prefetch_count,interleave_max)
        else:
            if "interleave"==img_mode:
                print(f"infer_mode is : interleave without image")
                text,image=infer_sensenova_interleave(model,prompt,cfg,steps,timestep_shift,img_cfg,images,cfg_interval,width,height,think_mode,seed,prefetch_count,interleave_max)
            else:
                print(f"infer_mode is : t2i")
                text,image=infer_sensenova_t2i(model,prompt,cfg,cfg_norm,steps,batch_size,timestep_shift,cfg_interval,width,height,seed,prefetch_count,think_mode,enhance,)

        if  prefetch_count is None:
            model.model.to(torch.device("cpu"))

        return io.NodeOutput(image,text)
