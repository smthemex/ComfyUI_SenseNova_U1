"""transformers_compat —— 唯一一处对 transformers 内部 API 的版本敏感出入口。

SenseNova-U1 的模型代码（modeling_qwen3*.py 里定义的类）是我们自己的实现，但借用了
transformers 的一小部分工具：mask 构造、RoPE 表、attention 后端、以及若干 typing/装饰器
辅助。这些工具的『导入路径、函数签名、甚至关键字名』在 transformers 4.x / 5.x / 未来 6.x
之间都会变，一旦用户装的是别的版本，原本硬编码 5.x 的代码就会直接报错。

为避免插件被钉死在某一版 transformers，这里在**导入时**集中完成所有版本敏感绑定：

  * 对每个符号，按已知的模块位置逐个 try/except 解析（多路径兜底）；
  * 调用约定不同处（如 ``create_causal_mask`` 的 ``inputs_embeds`` vs ``input_embeds``
    关键字）用运行时签名自省来选对名字；
  * 全部解析失败时，退回到本文件自带的最小实现，保证 import 永不崩。

modeling / configuration 模块**只允许从这里 import** 任何版本敏感符号，不要再写裸的
transformers 版本路径（如 ``transformers.masking_utils`` / ``transformers.modeling_layers``）。
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers


# --------------------------------------------------------------------------- #
# 版本信息（供调试 / 日志使用）
# --------------------------------------------------------------------------- #
TRANSFORMERS_VERSION = transformers.__version__


def _parse_version(vstr: str) -> Tuple[int, int, int]:
    nums: list[int] = []
    for part in str(vstr).split("."):
        try:
            nums.append(int(part))
        except ValueError:
            break
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])  # type: ignore[return-value]


TF_VERSION_TUPLE = _parse_version(TRANSFORMERS_VERSION)


# --------------------------------------------------------------------------- #
# 解析原语
# --------------------------------------------------------------------------- #
class _Missing:
    __repr__ = lambda self: "<missing>"  # noqa: E731


_MISSING = _Missing()


def _resolve(*specs: Tuple[str, str]):
    """按顺序尝试 (module, attr) 规格，返回第一个成功解析到的对象；都失败返回 _MISSING。"""
    for modname, attr in specs:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        obj = getattr(mod, attr, None)
        if obj is not None:
            return obj
    return _MISSING


def _fallback(obj: Any, default: Any) -> Any:
    return default if obj is _MISSING else obj


# 装饰器兜底：原版缺失时一律变成原样返回的空操作装饰器（可带参或不带参两种用法）
def _identity_decorator(maybe_target=None, **_kw):
    def _wrap(cls):
        return cls

    if maybe_target is not None:
        return _wrap(maybe_target)
    return _wrap


# --------------------------------------------------------------------------- #
# 1. typing / 装饰器辅助（5.x 多放在 transformers.utils / processing_utils 下）
# --------------------------------------------------------------------------- #
try:  # 5.x: transformers.processing_utils.Unpack
    from transformers.processing_utils import Unpack  # type: ignore
except Exception:  # 4.x / 3.11+ typing
    try:
        from typing_extensions import Unpack  # type: ignore
    except Exception:
        from typing import Unpack  # type: ignore


TransformersKwargs = _fallback(
    _resolve(("transformers.utils", "TransformersKwargs")),
    None,
)
if TransformersKwargs is None:
    from typing import TypedDict

    class TransformersKwargs(TypedDict, total=False):  # type: ignore[misc]
        """兜底：任意版本缺失 TransformersKwargs 时，给一个空 TypedDict，使 Unpack[...] 合法。"""

    TransformersKwargs = TransformersKwargs


FlashAttentionKwargs = _fallback(
    _resolve(("transformers.modeling_flash_attention_utils", "FlashAttentionKwargs")),
    None,
)
if FlashAttentionKwargs is None:
    from typing import TypedDict

    class FlashAttentionKwargs(TypedDict, total=False):  # type: ignore[misc]
        """兜底：FlashAttentionKwargs 缺失时给空 TypedDict。"""

    FlashAttentionKwargs = FlashAttentionKwargs


# auto_docstring 只用于给 transformers 官方文档站生成 docstring，运行期零作用。
# 但 transformers 5.x 的实现会在 import 时就跑校验：get_model_name() 按文件路径
# ".../models/<folder>/modeling_*.py" 的 <folder> 反推 model_type（本插件恰好是
# models/neo_unify/ → "neo-unify"），在 AUTO_CONFIGURATION_MAPPING 与
# HARDCODED_CONFIG_FOR_MODELS 里查不到就 print 一行 [ERROR]；同时它还会解析源码
# 把 forward 签名和 docstring 对账，插件新增的 indexes/cache_position/
# image_gen_indicators 没写进 docstring，又各 print 一行 [ERROR]。全是无关噪声，
# 故这里不做 _resolve，直接用空装饰器（原 docstring 原样保留）。
auto_docstring = _identity_decorator
can_return_tuple = _fallback(_resolve(("transformers.utils", "can_return_tuple")), _identity_decorator)
deprecate_kwarg = _fallback(
    _resolve(
        ("transformers.utils.deprecation", "deprecate_kwarg"),
        ("transformers.utils", "deprecate_kwarg"),
    ),
    _identity_decorator,
)
check_model_inputs = _fallback(
    _resolve(
        ("transformers.utils.generic", "check_model_inputs"),
        ("transformers.utils", "check_model_inputs"),
    ),
    _identity_decorator,
)
use_kernel_forward_from_hub = _fallback(
    _resolve(("transformers.integrations", "use_kernel_forward_from_hub")),
    _identity_decorator,
)


# --------------------------------------------------------------------------- #
# 2. RoPE 相关（5.x 在 modeling_rope_utils；4.x 在 models.qwen3.modeling_qwen3）
# --------------------------------------------------------------------------- #
ROPE_INIT_FUNCTIONS = _fallback(
    _resolve(
        ("transformers.modeling_rope_utils", "ROPE_INIT_FUNCTIONS"),
        ("transformers.models.qwen3.modeling_qwen3", "ROPE_INIT_FUNCTIONS"),
    ),
    {},
)
# 兜底为 {}：本模型 rope_type 恒为 "default"，走本地 _compute_default_rope_parameters，
# 不会索引 ROPE_INIT_FUNCTIONS；非 default 类型在老版本缺失时才会 KeyError（可接受）。

dynamic_rope_update = _fallback(
    _resolve(
        ("transformers.modeling_rope_utils", "dynamic_rope_update"),
        ("transformers.models.qwen3.modeling_qwen3", "dynamic_rope_update"),
    ),
    _identity_decorator,
)


# --------------------------------------------------------------------------- #
# 3. 头部 / 层基类（5.x 在 modeling_layers；4.x 部分在 modeling_utils）
# --------------------------------------------------------------------------- #
# 这些 GeneriXXX 基类仅用于补齐 HF 标准头类，推理（NEO 单体模型）实际用不到，
# 缺失时退回到空基类，保证类定义不崩、且 Qwen3PreTrainedModel 仍在 MRO 中。
_GenericHead = type("_GenericHead", (), {})
GenericForQuestionAnswering = _fallback(
    _resolve(("transformers.modeling_layers", "GenericForQuestionAnswering")), _GenericHead
)
GenericForSequenceClassification = _fallback(
    _resolve(("transformers.modeling_layers", "GenericForSequenceClassification")), _GenericHead
)
GenericForTokenClassification = _fallback(
    _resolve(("transformers.modeling_layers", "GenericForTokenClassification")), _GenericHead
)


class _GradientCheckpointingLayer(nn.Module):
    """兜底：4.x 的 GradientCheckpointingLayer 位于 modeling_utils，5.x 在 modeling_layers。
    都不存在时给一个最小实现，提供 decoder layer 需要的 gradient_checkpointing 属性。"""

    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def is_gradient_checkpointing(self) -> bool:
        return self.gradient_checkpointing


GradientCheckpointingLayer = _fallback(
    _resolve(
        ("transformers.modeling_layers", "GradientCheckpointingLayer"),
        ("transformers.modeling_utils", "GradientCheckpointingLayer"),
    ),
    _GradientCheckpointingLayer,
)


GenerationMixin = _fallback(_resolve(("transformers.generation", "GenerationMixin")), object)


# --------------------------------------------------------------------------- #
# 4. 因果 mask 构造：导入路径 + 关键字名都随版本不同
# --------------------------------------------------------------------------- #
create_causal_mask = _fallback(
    _resolve(
        ("transformers.masking_utils", "create_causal_mask"),          # 5.x
        ("transformers.modeling_attn_mask_utils", "create_causal_mask"),  # 4.x
    ),
    None,
)

if create_causal_mask is None:
    # 最后一招：自带一个最小 4D 加性因果 mask 实现，保证 import 与 forward 永不崩。
    def create_causal_mask(
        *,
        inputs_embeds=None,
        input_embeds=None,
        attention_mask=None,
        sliding_window=None,
        cache_position=None,
        device=None,
        dtype=None,
        config=None,
        **_kw,
    ):
        src = inputs_embeds if inputs_embeds is not None else input_embeds
        if src is None:
            raise ValueError("create_causal_mask fallback requires inputs_embeds/input_embeds")
        bsz, seqlen = src.shape[0], src.shape[1]
        dev = device or src.device
        dt = dtype or src.dtype
        causal = torch.full((seqlen, seqlen), float("-inf"), device=dev, dtype=dt)
        causal = torch.triu(causal, diagonal=1)
        return causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seqlen, seqlen)


def build_causal_mask_kwargs(
    *,
    inputs_embeds,
    attention_mask,
    sliding_window,
    cache_position,
    past_key_values,
    position_ids,
    config,
    device=None,
    dtype=None,
):
    """按当前 transformers 的 ``create_causal_mask`` 真实签名构造调用参数。

    transformers 把 token 嵌入的关键字从 ``input_embeds``（旧）改名成了
    ``inputs_embeds``（新）。这里用签名自省选对名字，避免硬编码。
    同时只保留对端真正接受的参数，未知版本也不会因多余关键字而崩。
    """
    params = set(inspect.signature(create_causal_mask).parameters)
    key = "inputs_embeds" if "inputs_embeds" in params else "input_embeds"
    payload = {
        "config": config,
        "attention_mask": attention_mask,
        "sliding_window": sliding_window,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "device": device,
        "dtype": dtype,
    }
    accepted = {k: v for k, v in payload.items() if k in params}
    accepted[key] = inputs_embeds
    return accepted


# --------------------------------------------------------------------------- #
# 5. attention 后端表（5.x 在 modeling_utils.ALL_ATTENTION_FUNCTIONS）
# --------------------------------------------------------------------------- #
def _compat_eager_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(value.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _compat_sdpa_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    attn_output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout, scale=scaling
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


_FALLBACK_ATTENTION = {
    "eager": _compat_eager_attention_forward,
    "sdpa": _compat_sdpa_attention_forward,
    "flash_attention_2": _compat_sdpa_attention_forward,
}

_ALL_ATTENTION_FUNCTIONS_REAL = _resolve(
    ("transformers.modeling_utils", "ALL_ATTENTION_FUNCTIONS"),
    ("transformers.models.qwen3.modeling_qwen3", "ALL_ATTENTION_FUNCTIONS"),
)
if _ALL_ATTENTION_FUNCTIONS_REAL is _MISSING:
    ALL_ATTENTION_FUNCTIONS: Dict[str, Callable] = dict(_FALLBACK_ATTENTION)
else:
    ALL_ATTENTION_FUNCTIONS = _ALL_ATTENTION_FUNCTIONS_REAL


# --------------------------------------------------------------------------- #
# 6. DynamicCache(config=...) 是 5.x 专属调用，4.x 不接受 config 关键字
# --------------------------------------------------------------------------- #
try:  # 稳定公共 API
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover
    DynamicCache = None  # type: ignore


def make_dynamic_cache(config=None):
    """版本无关地构造 DynamicCache：5.x 接受 config=，4.x 不接受。"""
    if DynamicCache is None:
        raise RuntimeError("transformers.cache_utils.DynamicCache 不可用")
    if config is None:
        return DynamicCache()
    try:
        return DynamicCache(config=config)
    except TypeError:
        return DynamicCache()


# --------------------------------------------------------------------------- #
# 7. config rope 参数归一化（5.x 合并成 rope_parameters；4.x 是 rope_theta/rope_scaling）
# --------------------------------------------------------------------------- #
def _build_rope_parameters(rope_theta, rope_scaling):
    """尝试构造 5.x 的 RopeParameters 数据类；任何版本差异导致构造失败都返回 None。"""
    for modname in (
        "transformers.modeling_rope_utils",
        "transformers.models.qwen3.modeling_qwen3",
    ):
        try:
            mod = importlib.import_module(modname)
            RopeParameters = getattr(mod, "RopeParameters", None)
            if RopeParameters is not None:
                return RopeParameters(
                    rope_type="default", rope_theta=rope_theta, rope_scaling=rope_scaling
                )
        except Exception:
            continue
    return None


def emit_rope_kwargs(config_cls, rope_theta, rope_scaling):
    """返回当前 transformers 版本下 ``config_cls.__init__`` 接受的 rope 关键字。

    模型真正的 RoPE 频率由我们自定义的 Qwen3RotaryEmbedding 读取
    ``config.rope_theta`` / ``config.rope_scaling`` 属性计算（与 transformers 版本无关），
    这里只是把父类的 rope 字段也填对，避免 ``_init_weights`` 之类的父类逻辑读到默认值。
    """
    init_params = set(inspect.signature(config_cls.__init__).parameters)
    if "rope_parameters" in init_params:
        rp = _build_rope_parameters(rope_theta, rope_scaling)
        if rp is not None:
            return {"rope_parameters": rp}
        return {}
    out: Dict[str, Any] = {}
    if "rope_theta" in init_params:
        out["rope_theta"] = rope_theta
    if "rope_scaling" in init_params:
        out["rope_scaling"] = rope_scaling
    return out


# --------------------------------------------------------------------------- #
# 8. 基座类（PreTrainedModel / PretrainedConfig）
#    这两个是模型类与配置类的根基类，最容易被未来版本挪动（例如 5.x 已把
#    PretrainedConfig 的某些再导出路径移除）。集中解析，保证 import 永不崩，
#    且未来 6.x 即便搬家也能被多路径捕获。
# --------------------------------------------------------------------------- #
PreTrainedModel = _fallback(
    _resolve(
        ("transformers.modeling_utils", "PreTrainedModel"),
        ("transformers.modeling_base", "PreTrainedModel"),
        ("transformers", "PreTrainedModel"),
    ),
    None,
)
if PreTrainedModel is None:
    # 最后一招兜底：给一个最小 nn.Module 基类，保证类定义与 super().__init__(config)
    # 不崩（权重经 nn.Module.load_state_dict 仍可加载；推理前向逻辑不依赖父类实现）。
    class PreTrainedModel(nn.Module):  # type: ignore[misc, valid-type]
        config_class = None
        _supports_flash_attn_2 = False
        supports_gradient_checkpointing = False

        def __init__(self, config=None):
            super().__init__()
            self.config = config

        def post_init(self):
            pass

        def _init_weights(self, module):
            pass


PretrainedConfig = _fallback(
    _resolve(
        ("transformers.configuration_utils", "PretrainedConfig"),
        ("transformers", "PretrainedConfig"),
    ),
    None,
)
if PretrainedConfig is None:
    # 最后一招兜底：最小配置基类，至少让 from_dict / 属性读写不崩。
    class PretrainedConfig:  # type: ignore[no-redef]
        model_type = "fallback"

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def to_dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        @classmethod
        def from_dict(cls, d, **kwargs):
            return cls(**d)


# --------------------------------------------------------------------------- #
# 调试信息：记录哪些符号走了兜底（仅 debug 级别，不干扰正常运行）
# --------------------------------------------------------------------------- #
def _engaged_fallbacks() -> Dict[str, str]:
    fb: Dict[str, str] = {}
    mapping = {
        "create_causal_mask": create_causal_mask,
        "ALL_ATTENTION_FUNCTIONS": ALL_ATTENTION_FUNCTIONS,
        "GradientCheckpointingLayer": GradientCheckpointingLayer,
        "Unpack": Unpack,
        "TransformersKwargs": TransformersKwargs,
        "FlashAttentionKwargs": FlashAttentionKwargs,
        "PreTrainedModel": PreTrainedModel,
        "PretrainedConfig": PretrainedConfig,
    }
    for name, obj in mapping.items():
        if obj is None or (name == "ALL_ATTENTION_FUNCTIONS" and obj is _FALLBACK_ATTENTION):
            fb[name] = "fallback"
    return fb


__all__ = [
    "TRANSFORMERS_VERSION",
    "TF_VERSION_TUPLE",
    "Unpack",
    "TransformersKwargs",
    "FlashAttentionKwargs",
    "auto_docstring",
    "can_return_tuple",
    "deprecate_kwarg",
    "check_model_inputs",
    "use_kernel_forward_from_hub",
    "ROPE_INIT_FUNCTIONS",
    "dynamic_rope_update",
    "GenericForQuestionAnswering",
    "GenericForSequenceClassification",
    "GenericForTokenClassification",
    "GradientCheckpointingLayer",
    "GenerationMixin",
    "create_causal_mask",
    "build_causal_mask_kwargs",
    "ALL_ATTENTION_FUNCTIONS",
    "make_dynamic_cache",
    "DynamicCache",
    "emit_rope_kwargs",
    "PreTrainedModel",
    "PretrainedConfig",
    "transformers_compat_info",
]

# transformers_compat_info 供外部在日志里打印当前解析情况
transformers_compat_info = {
    "transformers_version": TRANSFORMERS_VERSION,
    "tf_version_tuple": TF_VERSION_TUPLE,
    "engaged_fallbacks": _engaged_fallbacks(),
}
