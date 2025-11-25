# 1. What this model does

It is a $2\times$ super‑resolution model:
* Input: low‑resolution RGB image $H \times W$ → Output: $2H \times 2W$.

It expects:
* A float tensor in $[0, 1]$
* Shape $[B, 3, H, W]$

Internally, it downsamples by stride‑2 convolutions, so $H$ and $W$ should ideally be multiples of 4 (to avoid size mismatch). In the template below, we crop the image center to that.
This model is better to be used for upscaling textures but can be used for any type of image. also must be used for low quality image, do not use this on high quality ones.

Model Output Examples:

Terrain:
<img width="989" height="880" alt="1" src="https://github.com/user-attachments/assets/9fc90785-607a-46e0-bab2-de80beffeef9" />

Wood:
<img width="989" height="898" alt="2" src="https://github.com/user-attachments/assets/446228ca-1b9d-4bb8-af93-805474e6b9f5" />

Metal:
<img width="989" height="920" alt="3" src="https://github.com/user-attachments/assets/ecaa7df4-85f9-4c9b-8a8f-4395247fb901" />



# 2. Files you should have

The architecture definition (`class TextureEnhancementNet`) – extracted from your training script.
Example module file: `texture_model.py`:

```python
# texture_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels, growth_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2*growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3*growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4*growth_channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, channels, use_checkpointing=False):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels)
        self.rdb2 = ResidualDenseBlock(channels)
        self.rdb3 = ResidualDenseBlock(channels)
        self.use_checkpointing = use_checkpointing

    def forward(self, x):
        if self.use_checkpointing and self.training:
            out = checkpoint(self.rdb1, x, use_reentrant=False)
            out = checkpoint(self.rdb2, out, use_reentrant=False)
            out = checkpoint(self.rdb3, out, use_reentrant=False)
        else:
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
        return out * 0.2 + x

class TextureEnhancementNet(nn.Module):
    """
    2× super-resolution model (same as in training).
    """
    def __init__(self, num_channels=64, num_blocks=16, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        base_c = num_channels

        total_blocks = min(num_blocks, 18)
        n_enc1 = 2
        n_enc2 = 2
        n_bottleneck = 4
        n_dec2 = 2
        n_dec1 = 2
        n_extra = max(total_blocks - (n_enc1 + n_enc2 + n_bottleneck + n_dec2 + n_dec1), 2)

        self.conv_first = nn.Conv2d(3, base_c, 3, 1, 1)

        self.enc1_down = nn.Conv2d(base_c, base_c * 2, 3, 2, 1)
        self.enc1_body = nn.Sequential(*[
            RRDB(base_c * 2, use_checkpointing) for _ in range(n_enc1)
        ])

        self.enc2_down = nn.Conv2d(base_c * 2, base_c * 4, 3, 2, 1)
        self.enc2_body = nn.Sequential(*[
            RRDB(base_c * 4, use_checkpointing) for _ in range(n_enc2)
        ])

        self.bottleneck_body = nn.Sequential(*[
            RRDB(base_c * 4, use_checkpointing) for _ in range(n_bottleneck)
        ])

        self.dec2_up = nn.Conv2d(base_c * 4, base_c * 8, 3, 1, 1)
        self.dec2_body = nn.Sequential(*[
            RRDB(base_c * 2, use_checkpointing) for _ in range(n_dec2)
        ])

        self.dec1_up = nn.Conv2d(base_c * 2, base_c * 4, 3, 1, 1)
        self.dec1_body = nn.Sequential(*[
            RRDB(base_c, use_checkpointing) for _ in range(n_dec1)
        ])

        self.extra_body = nn.Sequential(*[
            RRDB(base_c, use_checkpointing) for _ in range(n_extra)
        ])

        self.up_final = nn.Conv2d(base_c, base_c * 4, 3, 1, 1)
        self.conv_last = nn.Conv2d(base_c, 3, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat0 = self.conv_first(x)

        feat1 = self.lrelu(self.enc1_down(feat0))
        feat1 = self.enc1_body(feat1)

        feat2 = self.lrelu(self.enc2_down(feat1))
        feat2 = self.enc2_body(feat2)

        bott = self.bottleneck_body(feat2)

        up2 = self.dec2_up(bott)
        up2 = F.pixel_shuffle(up2, 2)
        up2 = up2 + feat1
        up2 = self.dec2_body(up2)

        up1 = self.dec1_up(up2)
        up1 = F.pixel_shuffle(up1, 2)
        up1 = up1 + feat0
        up1 = self.dec1_body(up1)

        feat = self.extra_body(up1)

        feat_up = self.up_final(feat)
        feat_up = F.pixel_shuffle(feat_up, 2)
        out = self.conv_last(self.lrelu(feat_up))
        return torch.clamp(out, 0.0, 1.0)
```

Your trained weights, e.g. `Tx-Up.pth`.

# 3. Installation

In a fresh environment (local or Colab):

```bash
pip install torch torchvision pillow
```

# 4. Minimal usage template (single script)

Save the following as `use_texture_sr.py` (adjust paths and hyperparameters as needed):

```python
import os
import argparse

import torch
import numpy as np
from PIL import Image

from texture_model import TextureEnhancementNet


# ---------- Utilities ----------

def get_device(prefer_gpu: bool = True):
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    weights_path: str,
    device: torch.device,
    num_channels: int = 64,
    num_blocks: int = 16,
) -> torch.nn.Module:
    """
    Load TextureEnhancementNet with given hyperparameters and weights.
    """
    model = TextureEnhancementNet(
        num_channels=num_channels,
        num_blocks=num_blocks,
        use_checkpointing=False
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def crop_to_multiple_of_4(img: Image.Image) -> Image.Image:
    """
    Center-crop the image so width and height are multiples of 4.
    """
    w, h = img.size
    target_w = w - (w % 4)
    target_h = h - (h % 4)
    if target_w == w and target_h == h:
        return img

    left = (w - target_w) // 2
    top = (h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Convert PIL RGB image to normalized float tensor [1, 3, H, W] in [0, 1].
    """
    arr = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return t.to(device)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """
    Convert tensor [1, 3, H, W] in [0, 1] back to PIL RGB.
    """
    t = t.detach().cpu().clamp(0, 1)
    arr = t.squeeze(0).permute(1, 2, 0).numpy()  # [H, W, 3]
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def upscale_image(
    input_path: str,
    output_path: str,
    model: torch.nn.Module,
    device: torch.device
):
    """
    Load an image, run 2x SR with the model, and save the result.
    """
    # 1. Load and preprocess
    img = Image.open(input_path).convert("RGB")
    img = crop_to_multiple_of_4(img)
    lr_tensor = pil_to_tensor(img, device)

    # 2. Inference (no gradients)
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # 3. Convert and save
    sr_img = tensor_to_pil(sr_tensor)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sr_img.save(output_path)
    print(f"Saved upscaled image to: {output_path}")


# ---------- Command-line interface ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Use TextureEnhancementNet to perform 2x super-resolution."
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to trained weights (e.g., Tx-Up.pth)."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input image path."
    )
    parser.add_argument(
        "--output", type=str, default="upscaled.png",
        help="Output image path."
    )
    parser.add_argument(
        "--num-channels", type=int, default=64,
        help="num_channels used when training the model."
    )
    parser.add_argument(
        "--num-blocks", type=int, default=16,
        help="num_blocks (RRDB blocks) used when training."
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference (default: use GPU if available)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device(prefer_gpu=not args.cpu)
    print("Using device:", device)

    model = load_model(
        weights_path=args.weights,
        device=device,
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
    )

    upscale_image(
        input_path=args.input,
        output_path=args.output,
        model=model,
        device=device
    )


if __name__ == "__main__":
    main()
```

### How to run it

From a terminal / command prompt (in the same directory as `texture_model.py` and `use_texture_sr.py`):

```bash
python use_texture_sr.py \
  --weights Tx-Up.pth \
  --input path/to/your_lowres.png \
  --output path/to/upscaled.png \
  --num-channels 64 \
  --num-blocks 16
```

To force CPU:

```bash
python use_texture_sr.py --weights Tx-Up.pth --input in.png --output out.png --cpu
```

# 5. Using it programmatically (inside your own Python code)

If you prefer to call it from another script:

```python
import torch
from PIL import Image
from texture_model import TextureEnhancementNet
from use_texture_sr import load_model, pil_to_tensor, tensor_to_pil, crop_to_multiple_of_4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = load_model("Tx-Up.pth", device, num_channels=64, num_blocks=16)

# Load your PIL image
img = Image.open("input.png").convert("RGB")
img = crop_to_multiple_of_4(img)

# Convert to tensor
lr_tensor = pil_to_tensor(img, device)

# Inference
with torch.no_grad():
    sr_tensor = model(lr_tensor)


sr_img = tensor_to_pil(sr_tensor)
sr_img.save("upscaled.png")
```
