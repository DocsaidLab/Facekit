# PyFace

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href="https://github.com/DocsaidLab/PyFace/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/PyFace?color=ffa"></a>
    <a href="https://pypi.org/project/pyface_docsaid/"><img src="https://img.shields.io/pypi/v/pyface_docsaid.svg"></a>
    <a href="https://pypi.org/project/pyface_docsaid/"><img src="https://img.shields.io/pypi/dm/pyface_docsaid?color=9cf"></a>
</p>

## Introduction

PyFace is a Python library for face detection, face landmark, face detph, face recognition, etc.

<img src="docs/teaser.jpg" alt="PyFace Overview">

## Installation

### Requirements

- Python 3.10+

### Install from GitHub

```bash
pip install git+https://github.com/DocsaidLab/PyFace.git
```

## Usage

You can see [demo](demo) for more details.

### General usage

We recommend to use `FaceService` for integrating all face models.

```python
import capybara as cb
import pyface as pf

face_service = pf.FaceService(
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
@misc{lin2025pyface,
  author       = {Kun-Hsiang Lin},
  title        = {PyFace: An Integrated Python Package for Face Analysis},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/DocsaidLab/PyFace}}
}
```
