"""
Utility to visualize a saved PyTorch model checkpoint (.pth).

Outputs:
- model_summary.txt  (layerwise summary via torchinfo)
- model_graph.png    (autograd graph via torchviz, if available)
- model.onnx         (exported ONNX for Netron)

Usage:
  python visualize_model.py best_attentionlstm_20250907-091842.pth
"""

import argparse
import json
import os
from typing import Any, Dict

import torch
import numpy as np

# Local imports from this repo
from models import LSTMBaseline, GRUAlternative, TemporalConvNet
from advanced_models import AttentionLSTM, DeepLSTM, AdvancedTCN


MODEL_MAP = {
    'LSTMBaseline': LSTMBaseline,
    'GRUAlternative': GRUAlternative,
    'TemporalConvNet': TemporalConvNet,
    'AttentionLSTM': AttentionLSTM,
    'DeepLSTM': DeepLSTM,
    'AdvancedTCN': AdvancedTCN,
}


def load_metadata() -> Dict[str, Any]:
    meta_path = 'dataset_metadata_fixed.json'
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    # Fallback to non-fixed metadata if present
    meta_path = 'dataset_metadata.json'
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}


def infer_io_shapes(metadata: Dict[str, Any]) -> (int, int, int):
    # Default lookback and dims if metadata missing
    lookback = metadata.get('lookback_window', 36)
    feature_cols = metadata.get('feature_cols', [])
    target_cols = metadata.get('target_cols', [])
    input_size = len(feature_cols) if feature_cols else 11
    output_size = len(target_cols) if target_cols else 3
    return lookback, input_size, output_size


def _infer_attentionlstm_config_from_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Infer hidden_size and num_layers for AttentionLSTM from state_dict shapes."""
    try:
        # Hidden size: weight_hh_l0 has shape (4*hidden_size, hidden_size)
        hh0 = state_dict.get('lstm.weight_hh_l0')
        if hh0 is None:
            # Find a key containing lstm.weight_hh_l0
            for k, v in state_dict.items():
                if 'lstm.weight_hh_l0' in k and hasattr(v, 'shape'):
                    hh0 = v
                    break
        if hh0 is None:
            return {}
        hidden_size = hh0.shape[0] // 4
        # Num layers: count occurrences of weight_hh_l{idx}
        import re
        max_idx = -1
        for k in state_dict.keys():
            m = re.search(r"lstm\.weight_hh_l(\d+)", k)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
        num_layers = max_idx + 1 if max_idx >= 0 else 2
        return {'hidden_size': hidden_size, 'num_layers': num_layers}
    except Exception:
        return {}


def build_model(model_class: str, config: Dict[str, Any], input_size: int, output_size: int, state_dict: Dict[str, Any]):
    cls = MODEL_MAP.get(model_class)
    if cls is None:
        raise ValueError(f"Unsupported model class: {model_class}")

    # Normalize keys (some checkpoints might store slightly different names)
    cfg = dict(config or {})

    if model_class in ('LSTMBaseline', 'GRUAlternative'):
        kwargs = {
            'input_size': input_size,
            'output_size': output_size,
            'hidden_sizes': cfg.get('hidden_sizes', [128, 64]),
            'dropout_rate': cfg.get('dropout_rate', 0.2),
            'bidirectional': cfg.get('bidirectional', False),
        }
        return cls(**kwargs)

    if model_class == 'AttentionLSTM':
        # Try to infer from state_dict if config is missing or partial
        inferred = _infer_attentionlstm_config_from_state(state_dict)
        cfg = {**inferred, **cfg}
        kwargs = {
            'input_size': input_size,
            'hidden_size': cfg.get('hidden_size', 128),
            'num_layers': cfg.get('num_layers', 2),
            'output_size': output_size,
            'dropout_rate': cfg.get('dropout_rate', 0.2),
        }
        return cls(**kwargs)

    if model_class == 'TemporalConvNet':
        kwargs = {
            'input_size': input_size,
            'num_channels': cfg.get('num_channels', [64, 128, 64]),
            'output_size': output_size,
            'kernel_size': cfg.get('kernel_size', 3),
            'dropout': cfg.get('dropout', 0.2),
        }
        return cls(**kwargs)

    if model_class == 'AdvancedTCN':
        kwargs = {
            'input_size': input_size,
            'num_channels': cfg.get('num_channels', [64, 128, 256, 128, 64]),
            'output_size': output_size,
            'dropout': cfg.get('dropout', 0.2),
            'use_attention': cfg.get('use_attention', True),
        }
        return cls(**kwargs)

    if model_class == 'DeepLSTM':
        kwargs = {
            'input_size': input_size,
            'hidden_sizes': cfg.get('hidden_sizes', [256, 128, 64, 32]),
            'output_size': output_size,
            'dropout_rate': cfg.get('dropout_rate', 0.3),
            'layer_norm': cfg.get('layer_norm', True),
        }
        return cls(**kwargs)

    raise ValueError(f"Unhandled model class: {model_class}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt_path', help='Path to .pth checkpoint file')
    ap.add_argument('--seq-len', type=int, default=None, help='Override lookback/sequence length')
    args = ap.parse_args()

    # Robust checkpoint loading with PyTorch 2.6 safety defaults
    def _torch_load(path, weights_only=True):
        try:
            return torch.load(path, map_location='cpu', weights_only=weights_only)
        except TypeError:
            # Older PyTorch without weights_only arg
            return torch.load(path, map_location='cpu')

    # Try safe load first by allowlisting common numpy globals
    try:
        from torch.serialization import add_safe_globals
        # Allowlist numpy scalar and dtype for PyTorch 2.6 safe loader
        try:
            # Newer numpy path
            from numpy._core.multiarray import scalar as np_scalar
        except Exception:
            # Fallback path
            from numpy.core.multiarray import scalar as np_scalar
        import numpy as _np
        try:
            from numpy.dtypes import Float64DType as _Float64DType
        except Exception:
            _Float64DType = None
        extras = [np_scalar, _np.float32, _np.float64, _np.int64, _np.dtype]
        if _Float64DType is not None:
            extras.append(_Float64DType)
        add_safe_globals(extras)
    except Exception:
        pass

    try:
        ckpt = _torch_load(args.ckpt_path, weights_only=True)
    except Exception as e:
        print(f"Safe checkpoint load failed ({e}). Falling back to weights_only=False (trusted files only)...")
        ckpt = _torch_load(args.ckpt_path, weights_only=False)

    # Determine state_dict, class, and config
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        model_class = ckpt.get('model_class')
        model_config = ckpt.get('model_config', {})
    elif isinstance(ckpt, dict):
        # Might already be a pure state_dict
        state_dict = ckpt
        model_class = None
        model_config = {}
    else:
        raise RuntimeError('Unsupported checkpoint format')

    metadata = load_metadata()
    lookback, input_size, output_size = infer_io_shapes(metadata)
    if args.seq_len is not None:
        lookback = args.seq_len

    if not model_class:
        # Best effort guess from filename
        name = os.path.basename(args.ckpt_path).lower()
        if 'lstm' in name and 'attention' not in name:
            model_class = 'LSTMBaseline'
        elif 'gru' in name:
            model_class = 'GRUAlternative'
        elif 'attentionlstm' in name or 'attention' in name:
            model_class = 'AttentionLSTM'
        elif 'tcn' in name:
            model_class = 'TemporalConvNet'
        else:
            raise RuntimeError('Cannot infer model_class from checkpoint; please provide a checkpoint with model_class in it.')

    print(f"Model class: {model_class}")
    print(f"Input size: {input_size}, Output size: {output_size}, Seq len: {lookback}")
    print(f"Config (from checkpoint): {model_config}")

    model = build_model(model_class, model_config, input_size, output_size, state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    # Torchinfo summary
    try:
        from torchinfo import summary
        info = summary(model, input_size=(1, lookback, input_size))
        info_str = str(info)
        with open('model_summary.txt', 'w', encoding='utf-8') as f:
            f.write(info_str)
        print('Saved model_summary.txt')
    except Exception as e:
        print(f"torchinfo summary failed: {e}")

    # Torchviz graph
    try:
        x = torch.randn(1, lookback, input_size)
        y = model(x)
        try:
            from torchviz import make_dot
            dot = make_dot(y, params=dict(model.named_parameters()))
            dot.render('model_graph', format='png', cleanup=True)
            print('Saved model_graph.png')
        except Exception as e:
            print(f"torchviz graph failed: {e}")
    except Exception as e:
        print(f"Forward pass failed for graph: {e}")

    # ONNX export for Netron
    try:
        dummy = torch.randn(1, lookback, input_size)
        torch.onnx.export(
            model,
            dummy,
            'model.onnx',
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
            opset_version=14,
        )
        print('Saved model.onnx')
    except Exception as e:
        print(f"ONNX export failed: {e}")


if __name__ == '__main__':
    main()
