import numpy as np
cimport numpy as np
from libcpp cimport bool
cimport cython

np.import_array()

# cdef extern function definitions from C++
cdef extern from "rasterize.h":
    void _rasterize_triangles(
        float* vertices, int* triangles, float* depth_buffer, int* triangle_buffer, float* barycentric_weight,
        int ntri, int h, int w
    )

    void _rasterize(
        unsigned char* image, float* vertices, int* triangles, float* colors, float* depth_buffer,
        int ntri, int h, int w, int c, float alpha, bool reverse
    )

    void _get_tri_normal(float* tri_normal, float* vertices, int* triangles, int nver, bool norm_flg)
    void _get_ver_normal(float* ver_normal, float* tri_normal, int* triangles, int nver, int ntri)
    void _get_normal(float* ver_normal, float* vertices, int* triangles, int nver, int ntri)

# ====================
# Typed Memoryview API
# ====================

@cython.boundscheck(False)
@cython.wraparound(False)
def get_tri_normal(
    float[:, ::1] tri_normal,
    float[:, ::1] vertices,
    int[:, ::1] triangles,
    int ntri,
    bool norm_flg = False
):
    _get_tri_normal(&tri_normal[0, 0], &vertices[0, 0], &triangles[0, 0], ntri, norm_flg)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_ver_normal(
    float[:, ::1] ver_normal,
    float[:, ::1] tri_normal,
    int[:, ::1] triangles,
    int nver,
    int ntri
):
    _get_ver_normal(&ver_normal[0, 0], &tri_normal[0, 0], &triangles[0, 0], nver, ntri)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_normal(
    float[:, ::1] ver_normal,
    float[:, ::1] vertices,
    int[:, ::1] triangles,
    int nver,
    int ntri
):
    _get_normal(&ver_normal[0, 0], &vertices[0, 0], &triangles[0, 0], nver, ntri)


@cython.boundscheck(False)
@cython.wraparound(False)
def rasterize_triangles(
    float[:, ::1] vertices,
    int[:, ::1] triangles,
    float[:, ::1] depth_buffer,
    int[:, ::1] triangle_buffer,
    float[:, ::1] barycentric_weight,
    int ntri,
    int h,
    int w
):
    _rasterize_triangles(
        &vertices[0, 0], &triangles[0, 0], &depth_buffer[0, 0],
        &triangle_buffer[0, 0], &barycentric_weight[0, 0],
        ntri, h, w
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def rasterize(
    unsigned char[:, :, ::1] image,
    float[:, ::1] vertices,
    int[:, ::1] triangles,
    float[:, ::1] colors,
    float[:, ::1] depth_buffer,
    int ntri,
    int h,
    int w,
    int c,
    float alpha = 1,
    bool reverse = False
):
    _rasterize(
        &image[0, 0, 0], &vertices[0, 0], &triangles[0, 0],
        &colors[0, 0], &depth_buffer[0, 0],
        ntri, h, w, c, alpha, reverse
    )