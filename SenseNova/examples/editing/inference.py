from __future__ import annotations


import os
from typing import Sequence
import torch
from PIL import Image

from transformers import AutoConfig, AutoModel, AutoTokenizer
import gc   
# import sensenova_u1

from accelerate import init_empty_weights
from contextlib import AbstractContextManager

from ..utils import _streaming_model,_expert_streaming_ctx
from ...src.sensenova_u1.models.neo_unify.modeling_qwen3 import set_attn_backend
from safetensors.torch import load_file as st_load_file
from ...src.sensenova_u1.models.neo_unify.utils import load_image_native
from ...src.sensenova_u1.models.neo_unify.utils import smart_resize
from ...src.sensenova_u1.utils import (
    DEFAULT_IMAGE_PATCH_SIZE,
    InferenceProfiler,
    save_compare,
    
    load_and_merge_lora_weight_from_safetensors,
)

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

DEFAULT_SEED = 42
DEFAULT_SYSTEM_MESSAGE = """You are a multimodal assistant capable of reasoning with both text and images. You support two modes:\n\nThink Mode: When reasoning is needed, you MUST start with a <think></think> block and place all reasoning inside it. You MUST interleave text with generated images using tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and may be referenced in the final answer.\n\nNon-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any images naturally alongside the text.\n\nAfter the think block, always provide a concise, user-facing final answer. The answer may include text, images, or both. Match the user's language in both reasoning and the final answer."""

# Output H / W must be divisible by this (= patch_size * merge_size).
_IMAGE_GRID_FACTOR = DEFAULT_IMAGE_PATCH_SIZE

# aspect ratio ispreserved, total pixels are normalized to this target
DEFAULT_TARGET_PIXELS = 2048 * 2048
SUPPORTED_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "16:9": (2720, 1536),
    "9:16": (1536, 2720),
    "3:2": (2496, 1664),
    "2:3": (1664, 2496),
    "4:3": (2368, 1760),
    "3:4": (1760, 2368),
    "1:2": (1440, 2880),
    "2:1": (2880, 1440),
    "1:3": (1152, 3456),
    "3:1": (3456, 1152),
}
DEFAULT_WIDTH, DEFAULT_HEIGHT = SUPPORTED_RESOLUTIONS["1:1"]



def _denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _to_tensor(x: torch.Tensor) -> torch.Tensor:
   return _denorm(x.float()).permute(0, 2, 3, 1).cpu()


def _check_grid_divisible(width: int, height: int) -> None:
    if width % _IMAGE_GRID_FACTOR or height % _IMAGE_GRID_FACTOR:
        raise SystemExit(
            f"[editing] output resolution ({width}x{height}) must be a multiple "
            f"of {_IMAGE_GRID_FACTOR} on both axes (image-token grid factor)."
        )


def _resolve_output_size(
    input_images: Sequence[Image.Image],
    *,
    explicit: tuple[int, int] | None,
    target_pixels: int,
) -> tuple[int, int]:
    """Explicit (W, H) wins; else match the first input's aspect ratio and
    normalize the total pixel count to ``target_pixels``."""
    if explicit is not None:
        width, height = explicit
        _check_grid_divisible(width, height)
        return width, height

    w, h = input_images[0].size
    resized_h, resized_w = smart_resize(
        height=h,
        width=w,
        factor=_IMAGE_GRID_FACTOR,
        min_pixels=target_pixels,
        max_pixels=target_pixels,
    )
    return resized_w, resized_h


def _explicit_size_from_sample(sample: dict) -> tuple[int, int] | None:
    if "width" in sample and "height" in sample:
        return int(sample["width"]), int(sample["height"])
    return None



class SenseNovaU1Editing:
    """Thin wrapper calling ``model.it2i_generate`` on top of ``AutoModel``."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        checkpoint: str | None = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.model_path=model_path
        #check_checkpoint_compatibility(config)
        self.checkpoint = checkpoint
        self.repo=os.path.join(self.model_path,"SenseNova-U1-8B-MoT-SFT") if not "a3b" in self.checkpoint.lower() and  not "u1.5" in self.checkpoint.lower() else os.path.join(self.model_path,"SenseNova-U1.5-8B-MoT-Preview") if  "u1.5" in self.checkpoint.lower() else os.path.join(self.model_path,"SenseNova-U1-A3B-MoT-SFT")
        self.is_moe = "SenseNova-U1-A3B-MoT-SFT" in self.repo
        self.config = AutoConfig.from_pretrained(self.repo)
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo)
        self.model = None
        
    def _load_state_dict(self,lora_path=None):
        if self.model is not None:
            return  
        if self.checkpoint is not None:
            with init_empty_weights():
                self.model=AutoModel.from_config(self.config)
            if self.checkpoint.endswith(".gguf"):
                from ...convrot_loader import load_gguf_into_meta_model,load_gguf_lora
                info =load_gguf_into_meta_model(self.model, self.checkpoint,      
                                    device="cpu", dtype=torch.bfloat16)
                #print(info)
                if lora_path is not None:
                    lora_sd=st_load_file(lora_path)
                    load_gguf_lora(self.model,lora_sd)
                    del lora_sd
                # sd=load_gguf_checkpoint(self.checkpoint) 
                # #match_state_dict(self.model, sd,show_num=10)
                # lora_sd=st_load_file(lora_path) if lora_path is not None else None
                # set_gguf2meta_model(self.model,sd,self.dtype,torch.device("cpu"),lora_sd=lora_sd)
                # del sd
                self.model.eval() 
                # if lora_path is not None:
                #     del lora_sd
            else:
                sd=st_load_file(self.checkpoint)
                use_scale   = any(k.endswith(".scale_weight") for k in sd.keys())
                use_convrot = any(k.endswith(".comfy_quant") for k in sd.keys())
                if use_scale:
                    from ...fp8_scaled_loader import load_fp8_scaled_into_meta_model
                    info =load_fp8_scaled_into_meta_model(self.model, sd)
                    #print(info)
                elif use_convrot:
                    from ...convrot_loader import load_convrot_into_meta_model
                    info = load_convrot_into_meta_model(self.model, self.checkpoint,
                              )
                    #print(info)
                else:
                    self.model.load_state_dict(sd, strict=False, assign=True)
                    self.model = self.model.to(device=torch.device("cpu"),dtype=self.dtype)
                    self.model.eval()
                if lora_path is not None:
                    print(f"load lora {lora_path}")
                    lora_sd = st_load_file(lora_path)
                    # fp8-patched layers (FP8ScaledLinear): merge at de-quant time via
                    # the loader's LoRA buffers (mirrors load_gguf_lora).
                    if use_scale:
                        from ...fp8_scaled_loader import load_fp8_scaled_lora
                        load_fp8_scaled_lora(self.model, lora_sd)
                    elif use_convrot:
                        from ...convrot_loader import load_convrot_lora
                        load_convrot_lora(self.model, lora_sd)
                    else:
                        self.model = load_and_merge_lora_weight_from_safetensors(self.model, lora_path)
                    del lora_sd
                del sd
            gc.collect()
        else:
            raise ValueError("Checkpoint  is not provided for loading the model.")
            #self.model = AutoModel.from_pretrained(self.model_path, config=self.config, torch_dtype=self.dtype).to(device=torch.device("cpu")).eval()


    def _model_ctx(
        self,
        streaming_prefetch_count: int | None,
    ) -> AbstractContextManager:
        if streaming_prefetch_count is not None:
            if self.is_moe:
                return _expert_streaming_ctx(self.model,self.device,streaming_prefetch_count)  # 同步模式以减少内存占用
            else:
                return _streaming_model(
                    self.model,
                    layers_attr="language_model.model.layers",
                    target_device=self.device,
                    prefetch_count=streaming_prefetch_count,
                )

        return self.model
    @torch.inference_mode()
    def edit(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        image_size: tuple[int, int],
        cfg_scale: float = 4.0,
        img_cfg_scale: float = 1.0,
        cfg_norm: str = "none",
        timestep_shift: float = 3.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 50,
        batch_size: int = 1,
        think_mode = False,
        seed: int = 0,
        streaming_prefetch_count=1
    ) -> list[Image.Image]:
        
        if  streaming_prefetch_count is not None:
            with self._model_ctx(streaming_prefetch_count) as self.model:
                output = self.model.it2i_generate(
                    self.tokenizer,
                    prompt,
                    list(images),
                    image_size=image_size,
                    cfg_scale=cfg_scale,
                    img_cfg_scale=img_cfg_scale,
                    cfg_norm=cfg_norm,
                    timestep_shift=timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=num_steps,
                    batch_size=batch_size,
                    think_mode=think_mode,
                    seed=seed,
                )
        else:
             output = self.model.it2i_generate(
                self.tokenizer,
                prompt,
                list(images),
                image_size=image_size,
                cfg_scale=cfg_scale,
                img_cfg_scale=img_cfg_scale,
                cfg_norm=cfg_norm,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=num_steps,
                batch_size=batch_size,
                think_mode=think_mode,
                seed=seed,
            )
        return output
    @property
    def last_think_text(self) -> str:
        """Raw decoder output inside ``<think>...</think>`` (T2I think mode only)."""
        return self._last_think_text

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        image_size: tuple[int, int] = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
        cfg_scale: float = 4.0,
        cfg_norm: str = "none",
        timestep_shift: float = 3.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 50,
        batch_size: int = 1,
        seed: int = 0,
        think_mode: bool = False,
        streaming_prefetch_count=2
    ) -> list[Image.Image]:

        if  streaming_prefetch_count is not None: 
            with self._model_ctx(streaming_prefetch_count) as self.model:
                out = self.model.t2i_generate(
                    self.tokenizer,
                    prompt,
                    image_size=image_size,
                    cfg_scale=cfg_scale,
                    cfg_norm=cfg_norm,
                    timestep_shift=timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=num_steps,
                    batch_size=batch_size,
                    seed=seed,
                    think_mode=think_mode,
                )
        else:
            out = self.model.t2i_generate(
                self.tokenizer,
                prompt,
                image_size=image_size,
                cfg_scale=cfg_scale,
                cfg_norm=cfg_norm,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=num_steps,
                batch_size=batch_size,
                seed=seed,
                think_mode=think_mode,
            )
        if think_mode:
            tensor, think_text = out
            self._last_think_text = think_text
        else:
            tensor = out
            self._last_think_text = ""
        return _to_tensor(tensor)
    
    @torch.inference_mode()
    def answer(
        self,
        image,
        question: str,
        history: list | None = None,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        streaming_prefetch_count: int = 1,
    ) -> tuple[str, list]:
        pixel_values, grid_hw = load_image_native(image)
        pixel_values = pixel_values.to(self.device, dtype=self.model.dtype)
        grid_hw = grid_hw.to(self.device)

        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if do_sample:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = top_p
            if top_k is not None:
                generation_config["top_k"] = top_k
        if repetition_penalty is not None:
            generation_config["repetition_penalty"] = repetition_penalty
        if  streaming_prefetch_count is not None: 
            with self._model_ctx(streaming_prefetch_count) as self.model:
                response, updated_history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=history,
                    return_history=True,
                    grid_hw=grid_hw,
                )
        else:
            response, updated_history = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                history=history,
                return_history=True,
                grid_hw=grid_hw,
            )
        return response, updated_history


    @torch.inference_mode()
    def interleave_gen(
        self,
        prompt: str,
        input_images: Sequence[Image.Image] = (),
        image_size: tuple[int, int] = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
        cfg_scale: float = 4.0,
        img_cfg_scale: float = 1.0,
        max_images: int = 4,
        timestep_shift: float = 3.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 50,
        think_mode: bool = True,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        seed: int = 0,
        streaming_prefetch_count=1
    ) -> tuple[str, list[Image.Image]]:
        if input_images is None:
            input_images = ()
        if  streaming_prefetch_count is not None:
            with self._model_ctx(streaming_prefetch_count) as self.model:
                text, image_tensors = self.model.interleave_gen(
                    self.tokenizer,
                    prompt,
                    images=list(input_images),
                    image_size=image_size,
                    cfg_scale=cfg_scale,
                    img_cfg_scale=img_cfg_scale,
                    max_images=max_images,
                    timestep_shift=timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=num_steps,
                    system_message=system_message,
                    think_mode=think_mode,
                    seed=seed,
                )
        else:   
            text, image_tensors = self.model.interleave_gen(
                self.tokenizer,
                prompt,
                images=list(input_images),
                image_size=image_size,
                cfg_scale=cfg_scale,
                img_cfg_scale=img_cfg_scale,
                max_images=max_images,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=num_steps,
                system_message=system_message,
                think_mode=think_mode,
                seed=seed,
            )
        
        return text, torch.cat( [_to_tensor(i) for i in image_tensors], dim=0)




def  load_sensenova_model(model_path,device,repo,attn_backend,dtype=torch.bfloat16,lora_path=None):
    set_attn_backend(attn_backend)
    engine = SenseNovaU1Editing(repo, device, dtype,model_path)
    engine._load_state_dict(lora_path)
    return engine

def infer_sensenova_edit(engine,prompt,cfg_scale,cfg_norm,num_steps,batch_size,timestep_shift,img_cfg_scale,cfg_interval,width,height,images,target_pixels,seed,prefetch_count,think_mode=False):
    cfg_interval = tuple(cfg_interval)
    cli_explicit_size: tuple[int, int] | None = (width, height) if width is not None else None

    profiler = InferenceProfiler(enabled=True)
    #images = [_load_input_image(p) for p in args.image]
    #_maybe_warn_low_resolution_inputs(images, args.image, args.target_pixels)
    w, h = _resolve_output_size(
        images,
        explicit=cli_explicit_size,
        target_pixels=target_pixels,
    )
    # _set_seed(args.seed)
    
    with profiler.time_generate(w, h, batch_size):
        output = engine.edit(
            prompt,
            images,
            image_size=(w, h),
            cfg_scale=cfg_scale,
            img_cfg_scale=img_cfg_scale,
            cfg_norm=cfg_norm,
            timestep_shift=timestep_shift,
            cfg_interval=cfg_interval,
            num_steps=num_steps,
            batch_size=batch_size,
            think_mode = think_mode,
            seed=seed,
            streaming_prefetch_count=prefetch_count,
        )
        profiler.report()
    if think_mode:
        return _to_tensor(output[0]), output[1]
    return _to_tensor(output), "not think mode"


