__version__ = "0.1.2"
__date__    = "2026-03-24"
__author__  = "PolyU EIE4127 FYP"
__paper__   = "A Unified Approach for Target Direction Finding Based on CNNs (IEEE MLSP 2020)"
__doi__     = "10.1109/MLSP49062.2020.9231787"

# Changelog
# 0.1.2 (2026-03-24) - Smoke test 9/9 fix
#   - Fixed ReduceLROnPlateau verbose= arg removed in PyTorch 2.11
# 0.1.1 (2026-03-24) - Environment verified
#   - PyTorch 2.11.0+cu126 installed, CUDA=True, GPU=RTX 4090 D
#   - torchvision 0.26.0+cu126
#   - Fixed HDF5 chunk size bug (chunk must not exceed dataset size)
# 0.1.0 (2026-03-24) - Initial implementation
#   - TCA geometry: M=5, N=6 -> 12 sensors (positions verified against paper Fig.2)
#   - ResNeXt-50 backbone: 2ch input, FC(121)+Sigmoid
#   - BCELoss + AdamW trainer with early stopping + TensorBoard
#   - MUSIC + ESPRIT classical algorithms
