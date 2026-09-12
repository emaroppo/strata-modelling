"""Reading images into tensors: decoding, letterboxing, and the two datasets."""

import sys
from pathlib import Path

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from ...model import Example

# Video-extracted frames are occasionally cut short; decode what's there
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_rgb(path: str | Path, draft_size: int | None = None) -> Image.Image:
    try:
        img = Image.open(path)
        if draft_size is not None:
            # JPEG-only fast path: decode at reduced scale straight from the
            # DCT domain; PIL picks the smallest scale still >= draft_size
            img.draft("RGB", (draft_size, draft_size))
        return img.convert("RGB")
    except OSError as e:
        print(f"warning: unreadable image {path} ({e}), using black placeholder", file=sys.stderr)
        return Image.new("RGB", (256, 256))


class LetterboxSquash:
    """Resize to a square, splitting the aspect gap between distortion and padding.

    The image is squashed by at most ``max_distortion``; whatever aspect
    difference remains is letterboxed with black bands. For 16:9 input and
    max_distortion=1.4 the content fills ~79% of the square.
    """

    def __init__(self, size: int, max_distortion: float = 1.4):
        self.size = size
        self.max_distortion = max_distortion

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        aspect = w / h
        residual = max(aspect, 1 / aspect) / self.max_distortion
        if residual <= 1:
            content_w = content_h = self.size
        elif aspect > 1:
            content_w, content_h = self.size, round(self.size / residual)
        else:
            content_w, content_h = round(self.size / residual), self.size
        img = img.resize((content_w, content_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size))
        canvas.paste(img, ((self.size - content_w) // 2, (self.size - content_h) // 2))
        return canvas


class ImageDataset(Dataset):
    """Images with their encoded targets, for training and validation."""

    def __init__(
        self,
        samples: list[Example],
        classes: list[str],
        transform,
        draft_size: int | None = None,
        target_fn=None,
    ):
        self.samples = samples
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.draft_size = draft_size
        self.target_fn = target_fn

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = self.transform(load_rgb(sample.path, self.draft_size))
        return image, self.target_fn(sample.target.values, self.class_to_idx)


class InferenceDataset(Dataset):
    """Images alone, for prediction."""

    def __init__(self, paths: list[Path], transform, draft_size: int | None = None):
        self.paths = paths
        self.transform = transform
        self.draft_size = draft_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return self.transform(load_rgb(self.paths[idx], self.draft_size))
