import numpy
import perfplot

# NOTE: eisensum sometimes gives wrong result.
#       but sum gives accurate result
# Ref: https://stackoverflow.com/questions/18365073/why-is-numpys-einsum-faster-than-numpys-built-in-functions

def newaxis(data):
    A, b = data
    return A * b[:, numpy.newaxis]


def double_transpose(data):
    A, b = data
    return (A.T * b).T


def double_transpose_contiguous(data):
    A, b = data
    return numpy.ascontiguousarray((A.T * b).T)


def diag_dot(data):
    A, b = data
    return numpy.dot(numpy.diag(b), A)


def einsum(data):
    A, b = data
    return numpy.einsum('ij,i->ij', A, b)


perfplot.show(
    setup=lambda n: (numpy.random.rand(n, n), numpy.random.rand(n)),
    kernels=[
        newaxis, double_transpose, double_transpose_contiguous, diag_dot,
        einsum
        ],
    n_range=[2**k for k in range(10)],
    logx=True,
    logy=True,
    xlabel='len(A), len(b)'
    )
