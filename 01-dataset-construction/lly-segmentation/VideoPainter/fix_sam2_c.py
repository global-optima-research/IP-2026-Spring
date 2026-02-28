# -*- coding: utf-8 -*-
"""Replace SAM2 C extension with pure Python implementation"""

path = '/data/liuluyan/VideoPainter/app/sam2/utils/misc.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old = '''def get_connected_components(mask):
    """
    Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W).

    Inputs:
    - mask: A binary mask tensor of shape (N, 1, H, W), where 1 is foreground and 0 is
            background.

    Outputs:
    - labels: A tensor of shape (N, 1, H, W) containing the connected component labels
              for foreground pixels and 0 for background pixels.
    - counts: A tensor of shape (N, 1, H, W) containing the area of the connected
              components for foreground pixels and 0 for background pixels.
    """
    from sam2 import _C

    return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())'''

new = '''def get_connected_components(mask):
    """
    Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W).
    Pure Python implementation using scipy (no C extension needed).
    """
    from scipy import ndimage
    import numpy as np

    mask_np = mask.cpu().numpy().astype(np.uint8)
    N, C, H, W = mask_np.shape
    labels_out = np.zeros_like(mask_np, dtype=np.int32)
    counts_out = np.zeros_like(mask_np, dtype=np.int32)

    for n in range(N):
        for c in range(C):
            labeled, num_features = ndimage.label(mask_np[n, c])
            labels_out[n, c] = labeled
            for i in range(1, num_features + 1):
                area = (labeled == i).sum()
                counts_out[n, c][labeled == i] = area

    labels_tensor = torch.from_numpy(labels_out).to(mask.device)
    counts_tensor = torch.from_numpy(counts_out).to(mask.device)
    return labels_tensor, counts_tensor'''

if old in content:
    content = content.replace(old, new)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed! Replaced C extension with scipy implementation.")
else:
    print("ERROR: Could not find the function to replace")
    idx = content.find("def get_connected_components")
    if idx >= 0:
        print("Found at index", idx)
        print("Context:", repr(content[idx:idx+200]))
