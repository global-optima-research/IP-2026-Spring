"""Patch CausVid attention.py to add PyTorch SDPA fallback when flash_attn is not available."""
import sys

ATTN_FILE = "/data/chenqingzhan/CausVid/causvid/models/wan/wan_base/modules/attention.py"

with open(ATTN_FILE, "r") as f:
    content = f.read()

OLD = "    else:\n        assert FLASH_ATTN_2_AVAILABLE\n        x = flash_attn.flash_attn_varlen_func("

NEW = """    else:
        # Fallback to PyTorch SDPA when flash_attn is not available
        if not FLASH_ATTN_2_AVAILABLE:
            q_unflat = q.unflatten(0, (b, lq))
            k_unflat = k.unflatten(0, (b, lk))
            v_unflat = v.unflatten(0, (b, lk))
            q_sdpa = q_unflat.transpose(1, 2)
            k_sdpa = k_unflat.transpose(1, 2)
            v_sdpa = v_unflat.transpose(1, 2)
            x = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                is_causal=causal, dropout_p=dropout_p,
                scale=softmax_scale
            )
            x = x.transpose(1, 2).contiguous()
            return x.type(out_dtype)
        x = flash_attn.flash_attn_varlen_func("""

if OLD not in content:
    print("ERROR: Could not find target string in attention.py")
    print("File may have already been patched or has unexpected format")
    sys.exit(1)

content = content.replace(OLD, NEW)

with open(ATTN_FILE, "w") as f:
    f.write(content)

print("SUCCESS: attention.py patched with PyTorch SDPA fallback")
