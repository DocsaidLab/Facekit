# Facekit

## Introduction

Facekit is a Python library for face detection, face landmark, face detph, face recognition, etc.

## Installation

### Requirements

- Python 3.10+

### Install from GitHub

```bash
tag=v0.2.0
pip install git+https://git@github.com/DocsaidLab/Facekit.git@$tag
```

## Usage

You can see [demo](demo) for more details.

### General usage

We recommend to use `FaceService` for integrating all face models.

```python
import capybara as cb
import facekit as fk

face_service = fk.FaceService(
    batch_size=1,
    enable_recognition=True,
    enable_depth=True,
    enable_landmark=True,
    face_bank='path/to/face_bank',
    recog_level='High',
)
img = cb.imread('path/to/image')
faces_on_img = face_service([img])[0]
# Plotted faces on image
cb.imwrite('path/to/output', faces_on_img.gen_info_img())
```

## Citation
```bibtex
@misc{lin2025facekit,
  author = {Kun-Hsiang Lin},
  title = {Facekit: A face tool kit for easy face processing},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/DocsaidLab/Facekit},
  note = {GitHub repository}
}
```
