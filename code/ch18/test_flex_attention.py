import arch_config  # noqa: F401 - Configure Blackwell optimizations
#!/usr/bin/env python3
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Try to import flex_attention
try:
    from torch.nn.attention import flex_attention
    print(" FlexAttention imported successfully!")
    print("Type:", type(flex_attention))
    print("Available methods:", [x for x in dir(flex_attention) if not x.startswith('_')])
except ImportError as e:
    print(" ImportError:", e)
except AttributeError as e:
    print(" AttributeError:", e)
except Exception as e:
    print(" Unexpected error:", e)

# Check what's available in torch.nn.attention
print("\nAvailable in torch.nn.attention:")
import torch.nn.attention as attention
for item in dir(attention):
    if not item.startswith('_'):
        print(f"  - {item}")
