# PyFace

[![license](https://img.shields.io/badge/license-Apache%202-dfd.svg)](./LICENSE)
[![python](https://img.shields.io/badge/python-3.10+-aff.svg)](./pyproject.toml)
[![release](https://img.shields.io/github/v/release/DocsaidLab/PyFace?color=ffa)](https://github.com/DocsaidLab/PyFace/releases)
[![pypi](https://img.shields.io/pypi/v/pyface-docsaid.svg)](https://pypi.org/project/pyface-docsaid/)
[![downloads](https://img.shields.io/pypi/dm/pyface-docsaid?color=9cf)](https://pypi.org/project/pyface-docsaid/)

## Introduction

PyFace is a Python library for face detection, face landmark, face depth, face recognition, etc.

![PyFace Overview](https://media.githubusercontent.com/media/DocsaidLab/PyFace/refs/heads/main/docs/teaser.jpg)

## Installation

### Requirements

- Python 3.10+

### Install via pypi

```bash
pip install -U pip wheel
pip install setuptools # for python 3.12
pip install pyface-docsaid
```

### Install via GitHub

```bash
pip install -U pip wheel
pip install setuptools # for python 3.12
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
    detect_kwargs={"gpu_id": 0, "backend": "cuda"}, # if you want to use GPU on detection
    landmark_kwargs={"backend": "cpu"}, # if you want to use CPU on landmark
    ...
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
