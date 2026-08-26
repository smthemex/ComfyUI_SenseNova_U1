from __future__ import annotations
import functools
import itertools
import logging
from typing import Any
import torch.nn.functional as F
import torch
from torch import nn
from .src.sensenova_u1.models.neo_unify.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
logger = logging.getLogger(__name__)

class ExpertOffloadMoEBlock(nn.Module):
    """
    同步卸载，支持自定义缓存容量：在 GPU 上最多同时保留 `cache_capacity` 个专家。
    gate 常驻 GPU，专家按需同步加载/卸载，采用 LRU 淘汰策略。

    指针式（零拷贝）实现：每个专家的权威 CPU 权重引用在 __init__ 保存，
    load 时 ``.to(device)`` 新建 GPU 张量（源稳定，不丢原始权重），
    unload 时直接将 ``param.data`` 指回已保存的 CPU 引用（不新建 CPU 副本），
    从而避免每次切换专家都产生无谓的 CPU 副本复制与 CUDA 分配器碎片累积。
    """
    def __init__(self, moe_block, target_device: torch.device, cache_capacity: int = 1):
        super().__init__()
        self.moe_block = moe_block
        self.target_device = target_device
        self.cache_capacity = max(1, cache_capacity)  # 至少为1
        # gate 常驻 GPU
        self.moe_block.gate.to(target_device)
        self._resident_experts: list[int] = []
        # 每个专家的权威 CPU 权重引用（指针式卸载的基础）
        self._expert_cpu_refs: dict[int, list[torch.Tensor]] = {}
        for eid, expert in enumerate(self.moe_block.experts):
            refs = []
            for t in itertools.chain(expert.parameters(), expert.buffers()):
                d = t.data
                # 若专家权重已驻留在 GPU（上一轮残留），先搬回 CPU 再保存引用，
                # 确保 _expert_cpu_refs 永远是真正的 CPU 张量，unload 零拷贝指回才正确。
                refs.append(d if d.device.type == 'cpu' else d.to('cpu'))
            self._expert_cpu_refs[eid] = refs

    # ----- 代理原始属性 -----
    @property
    def num_experts(self):
        return self.moe_block.num_experts

    @property
    def top_k(self):
        return self.moe_block.top_k

    @property
    def norm_topk_prob(self):
        return self.moe_block.norm_topk_prob

    # ----- LRU 管理 -----
    def _touch(self, eid: int):
        """将专家 eid 移到 LRU 队列末尾（最近使用）。"""
        if eid in self._resident_experts:
            self._resident_experts.remove(eid)
        self._resident_experts.append(eid)

    def _evict_lru(self):
        """如果驻留数量超过 cache_capacity，卸载最久未使用的专家。"""
        while len(self._resident_experts) >= self.cache_capacity:
            victim = self._resident_experts.pop(0) # 最久未使用
            self._unload_expert(victim)

    def _load_expert(self, eid: int):
        """确保专家 eid 驻留在 GPU。若缓存已满，先淘汰最久未使用的。"""
        if eid in self._resident_experts:
            self._touch(eid)  # 命中，只更新时间戳
            return

        # 淘汰直至有空间
        self._evict_lru()

        # 加载新专家：以权威 CPU 引用为源，新建 GPU 张量
        expert_module = self.moe_block.experts[eid]
        refs = self._expert_cpu_refs[eid]
        params = list(itertools.chain(expert_module.parameters(), expert_module.buffers()))
        for param, ref in zip(params, refs):
            param.data = ref.to(self.target_device)
        self._resident_experts.append(eid)

    def _unload_expert(self, eid: int):
        """将指定专家从 GPU 移回 CPU（零拷贝：直接指回权威 CPU 引用）。"""
        expert_module = self.moe_block.experts[eid]
        refs = self._expert_cpu_refs[eid]
        params = list(itertools.chain(expert_module.parameters(), expert_module.buffers()))
        for param, ref in zip(params, refs):
            param.data = ref

    def _unload_all(self):
        """卸载所有驻留专家，释放显存。"""
        for eid in list(self._resident_experts):
            self._unload_expert(eid)
        self._resident_experts.clear()

   
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.moe_block.gate(flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat.dtype)

        active_experts = selected_experts.unique().tolist()

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        output = torch.zeros_like(flat)
       
        for eid in active_experts:
            
            self._load_expert(eid)

            idx, top_x = torch.where(expert_mask[eid])
            if top_x.numel() == 0:
                continue

            expert_module = self.moe_block.experts[eid]
            current_state = flat.index_select(0, top_x)
            expert_out = expert_module(current_state) * routing_weights[top_x, idx, None]
            output.index_add_(0, top_x, expert_out.to(flat.dtype))


        # ---- forward 结束，清空所有驻留专家（可选） ----
        #self._unload_all()

        return output.view(*orig_shape)

class ExpertStreamingWrapper(nn.Module):
    """
    同步版模型包装器：替换所有 MoE 块为 ExpertOffloadMoEBlock，
    将非 MoE 参数移到 GPU，MoE 专家按需同步加载。
    """

    def __init__(self, model: nn.Module, target_device: torch.device,cache_capacity: int = 1):
        super().__init__()
        self._model = model
        self._target_device = target_device
        self._cache_capacity = cache_capacity
        self._replace_map: dict[int, tuple[nn.Module, str, nn.Module]] = {}
        # 非层参数的 CPU 原始权重引用（指针式搬运用）
        self._non_layer_cpu_refs: dict[int, torch.Tensor] = {}

        expert_param_ids: set[int] = set()
        for module in model.modules():
            if isinstance(module, Qwen3MoeSparseMoeBlock):
                for expert in module.experts:
                    for p in expert.parameters():
                        expert_param_ids.add(id(p))
                    for b in expert.buffers():
                        expert_param_ids.add(id(b))
                for p in module.gate.parameters():
                    expert_param_ids.add(id(p))
                for b in module.gate.buffers():
                    expert_param_ids.add(id(b))


        self._replace_moe_blocks(model, target_device)

        # 指针式搬运非层参数：首次新建 GPU 并保存 CPU 引用，之后同设备复用
        def _ensure(param: torch.Tensor) -> None:
            pid = id(param)
            if pid not in self._non_layer_cpu_refs:
                d = param.data
                # 若权重已驻留在 GPU（上一轮残留），先搬回 CPU 再保存引用，
                # 确保 _non_layer_cpu_refs 永远是真正的 CPU 张量，指针式复用才正确。
                self._non_layer_cpu_refs[pid] = d if d.device.type == 'cpu' else d.to('cpu')
            param.data = self._non_layer_cpu_refs[pid].to(self._target_device)

        for p in model.parameters():
            if id(p) not in expert_param_ids:
                _ensure(p)
        for b in model.buffers():
            if id(b) not in expert_param_ids:
                _ensure(b)

    def _replace_moe_blocks(self, module: nn.Module, target_device: torch.device):
        replacements = []
        for name, child in module.named_children():
            if isinstance(child, Qwen3MoeSparseMoeBlock):
                replacements.append((name, child))
            else:
                self._replace_moe_blocks(child, target_device)
        for name, original in replacements:
            wrapper = ExpertOffloadMoEBlock(original, target_device, self._cache_capacity)
            setattr(module, name, wrapper)
            self._replace_map[id(wrapper)] = (module, name, original)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def teardown(self):
        for wrapper_id, (parent, name, original) in self._replace_map.items():
            setattr(parent, name, original)
        self._model.to('cpu')
        torch.cuda.synchronize(self._target_device)
        self._replace_map.clear()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
        
# edit from LayerStreamingWrapper from https://github.com/Lightricks/LTX-2

class _SimpleLayerStore:
    """简化版层存储，支持按需加载和立即释放（指针式：unload 零拷贝）"""

    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)
        self._resident: set[int] = set()

        # 保留CPU端的原始参数引用（权威副本，永不被覆盖）
        self._cpu_params: list[dict[str, torch.Tensor]] = []
        for layer in layers:
            cpu_copy = {}
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                if tensor is None:
                    continue
                # 若层张量已驻留在 GPU（如上一轮常驻层残留），先搬回 CPU 再保存引用，
                # 确保 _cpu_params 永远是真正的 CPU 张量，unload 时零拷贝指回才正确。
                t = tensor.data
                if t.device.type != 'cpu':
                    t = t.to('cpu')
                cpu_copy[name] = t
            self._cpu_params.append(cpu_copy)

    def set_resident(self, resident: set[int]) -> None:
        self._resident = resident

    def is_resident(self, idx: int) -> bool:
        return idx in self._resident

    def load_layer_to_gpu(self, idx: int, layer: nn.Module, force: bool = False) -> None:
        """将指定层加载到GPU"""
        if idx in self._resident and not force:
            return
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if param is None:
                continue
            if name in self._cpu_params[idx]:
                param.data = self._cpu_params[idx][name].to(self.target_device)

    def unload_layer_from_gpu(self, idx: int, layer: nn.Module) -> None:
        """将指定层从GPU卸载回CPU（零拷贝：直接指回权威CPU引用）"""
        if idx in self._resident:
            return
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if param is None:
                continue
            if name in self._cpu_params[idx]:
                param.data = self._cpu_params[idx][name]  # 恢复为CPU副本（同一引用，不新建）

def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    """Resolve a dotted attribute path like ``'model.language_model.layers'``."""
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj

class SimpleLayerStreamingWrapper(nn.Module):
    """简化版层流式处理包装器（指针式，对齐 FireRedAudio fast）

    - 常驻层：``active_count >= 1`` 时前 ``active_count`` 层一次性加载到 GPU 并
      不注册卸载钩子（不参与流式循环）；``active_count == 0`` 时全部层逐层流式。
    - 层权重：``_SimpleLayerStore`` 保存权威 CPU 引用，load 时 ``.to(gpu)`` 新建
      GPU 张量、unload 时零拷贝指回 CPU 引用（不新建）。
    - 非层参数（dit/patch/embed/lm_head 等）：指针式复用，首次 ``.to(gpu)`` 新建
      并保存 CPU 引用，之后 ``.to(gpu)`` 同设备返回自身（不复制），避免每次推理
      反复重建导致 CUDA 分配器碎片累积（消除"冷启动快、连续推理变慢"）。
    - post_hook 同步事件后主动 ``empty_cache()``，及时归还单层临时显存。
    """
    
    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        active_count: int = 0,  # 常驻层数（>=1 常驻前 N 层；0 表示全流式）
    ) -> None:
        super().__init__()
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
        self._active_count = active_count
        self._store = _SimpleLayerStore(self._layers, self._target_device)

        # 非层参数的 CPU 原始权重引用（指针式搬运用）
        self._non_layer_cpu_refs: dict[int, torch.Tensor] = {}

        n = len(self._layers)
        if active_count is not None and active_count >= 0:
            resident_count = min(active_count, n)
            self._resident = set(range(resident_count))
        else:
            self._resident = set()
        self._store.set_resident(self._resident)

        # 将非层参数移到 GPU（指针式常驻）
        self._move_non_layer_params_to_gpu()

        # 常驻层强制加载到 GPU（不注册卸载钩子）
        for idx in sorted(self._resident):
            self._store.load_layer_to_gpu(idx, self._layers[idx], force=True)

        # 注册钩子（仅非常驻层）
        self._register_simple_hooks()
    
    def _move_non_layer_params_to_gpu(self) -> None:
        """指针式搬运非层参数到 GPU"""
        layer_tensor_ids = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        def _ensure(param: torch.Tensor) -> None:
            pid = id(param)
            if pid not in self._non_layer_cpu_refs:
                self._non_layer_cpu_refs[pid] = param.data  # 保存权威 CPU 引用
            # 若已在目标设备，.to() 同设备返回自身（不复制/不新建）
            param.data = self._non_layer_cpu_refs[pid].to(self._target_device)

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                _ensure(p)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                _ensure(b)
    
    def _register_simple_hooks(self) -> None:
        """注册简单的加载/释放钩子（仅非常驻层）"""
        idx_map = {id(layer): idx for idx, layer in enumerate(self._layers)}
        
        def _pre_hook(module: nn.Module, input, *, idx: int):
            # 加载当前层到GPU
            self._store.load_layer_to_gpu(idx, module)
            # 记录流，防止内存被提前回收
            for param in itertools.chain(module.parameters(), module.buffers()):
                param.data.record_stream(torch.cuda.current_stream(self._target_device))
        
        def _post_hook(module: nn.Module, input, output, *, idx: int):
            # 等待该层在 GPU 上的计算真正完成，再卸载回 CPU
            event = getattr(self, "_events", {}).get(idx)
            if event is not None:
                event.record(torch.cuda.current_stream(self._target_device))
                event.synchronize()
            # 处理完后立即将层移回CPU（零拷贝指回 CPU 引用）
            self._store.unload_layer_from_gpu(idx, module)
            # 强制清空 CUDA 缓存分配器中的空闲块，防止异步计算与快速卸载
            # 导致大量已释放显存被缓存，从而在 nvidia-smi 中显示偏高
            torch.cuda.empty_cache()
        
        for layer in self._layers:
            idx = idx_map[id(layer)]
            if idx in self._resident:
                continue  # 常驻层不注册卸载钩子
            pre_hook = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            post_hook = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
    
    def to(self, *args: Any, **kwargs: Any) -> "SimpleLayerStreamingWrapper":
        """覆盖 nn.Module.to，走指针式搬运而非反复 .to() 新建。

        - to("cpu")：仅将非常驻层指回 CPU 引用以释放其 GPU 张量；
          非层参数保持 GPU 常驻（不卸载），以便下一轮推理复用同一 GPU 张量，
          从根本上避免 CUDA 分配器碎片累积。
        - 其他（如 to(cuda)）：确保非层参数在 GPU（指针式复用）。
        """
        target = None
        if args:
            a = args[0]
            if isinstance(a, (str, torch.device)):
                target = torch.device(a)
        if target is not None and target.type == "cpu":
            for idx in range(len(self._layers)):
                if not self._store.is_resident(idx):
                    self._store.unload_layer_from_gpu(idx, self._layers[idx])
            # 非层参数保持 GPU 常驻，不卸载
        else:
            self._move_non_layer_params_to_gpu()
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """代理属性访问到原始模型"""
        try:
            # 首先尝试从包装器自身获取属性
            return super().__getattr__(name)
        except AttributeError:
            # 如果失败，则从原始模型获取
            return getattr(self._model, name)
    


