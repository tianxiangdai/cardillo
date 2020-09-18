import numpy as np
from cardillo.discretization.indexing import flat2D, flat3D, split2D, split3D

def lagrange_basis1D(degree, xi, derivative=1):
    p = degree

    if not hasattr(xi, '__len__'):
        xi = np.array([xi])
 
    n = sum([1 for d in range(derivative + 1)])
    NN = np.zeros((k, p+1, n))
    #TODO: make seperate 1D Basis function with second derrivative
    Nxi = np.transpose(np.array(Lagrange_basis(p, xi)),(1,2,0))

    NN = Nxi[:n]

    return NN

def lagrange_basis2D(degrees, xis, derivative=1):
    p, q = degrees
    xi, eta = xis
    p1q1 = (p + 1) * (q + 1)

    if not hasattr(xi, '__len__'):
        xi = np.array([xi])
    if not hasattr(eta, '__len__'):
        eta = np.array([eta])

    k = len(xi)
    l = len(eta)
    kl = k * l
 
    n = sum([2**d for d in range(derivative + 1)])
    NN = np.zeros((kl, p1q1, n))
    #TODO: make seperate 1D Basis function with second derrivative
    Nxi = np.transpose(np.array(Lagrange_basis(p, xi)),(1,2,0))
    Neta = np.transpose(np.array(Lagrange_basis(q, eta)),(1,2,0))

    for i in range(kl):
        ik, il = split2D(i, (k, ))

        for a in range(p1q1):
            a_xi, a_eta = split2D(a, (p + 1, ))
            NN[i, a, 0] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 0]

            if derivative > 0:
                NN[i, a, 1] = Nxi[ik, a_xi, 1] * Neta[il, a_eta, 0] 
                NN[i, a, 2] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 1] 
                if derivative > 1:
                    raise NotImplementedError('...')
                    NN[i, a, 3] = Nxi[ik, a_xi, 2] * Neta[il, a_eta, 0]
                    NN[i, a, 4] = Nxi[ik, a_xi, 1] * Neta[il, a_eta, 1]
                    NN[i, a, 6] = NN[i, a, 4]
                    NN[i, a, 5] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 2] 

    return NN

def lagrange_basis3D(degrees, xis, derivative=1):
    p, q, r = degrees
    xi, eta, zeta = xis
    p1q1r1 = (p + 1) * (q + 1) * (r + 1)

    if not hasattr(xi, '__len__'):
        xi = np.array([xi])
    if not hasattr(eta, '__len__'):
        eta = np.array([eta])
    if not hasattr(zeta, '__len__'):
        zeta = np.array([zeta])

    k = len(xi)
    l = len(eta)
    m = len(zeta)
    klm = k * l * m
 
    n = sum([3**d for d in range(derivative + 1)])
    NN = np.zeros((klm, p1q1r1, n))
    #TODO: make seperate 1D Basis function with second derrivative
    Nxi = np.transpose(np.array(Lagrange_basis(p, xi)),(1,2,0))
    Neta = np.transpose(np.array(Lagrange_basis(q, eta)),(1,2,0))
    Nzeta = np.transpose(np.array(Lagrange_basis(r, zeta)),(1,2,0))

    for i in range(klm):
        ik, il, im = split3D(i, (k, l))

        for a in range(p1q1r1):
            a_xi, a_eta, a_zeta = split3D(a, (p + 1, q + 1))
            NN[i, a, 0] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 0] * Nzeta[im, a_zeta, 0]

            if derivative > 0:
                NN[i, a, 1] = Nxi[ik, a_xi, 1] * Neta[il, a_eta, 0] * Nzeta[im, a_zeta, 0]
                NN[i, a, 2] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 1] * Nzeta[im, a_zeta, 0]
                NN[i, a, 3] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 0] * Nzeta[im, a_zeta, 1]
                if derivative > 1:
                    raise NotImplementedError('...')
                    NN[i, a, 4] = Nxi[ik, a_xi, 2] * Neta[il, a_eta, 0] * Nzeta[im, a_zeta, 0]
                    NN[i, a, 5] = Nxi[ik, a_xi, 1] * Neta[il, a_eta, 1] * Nzeta[im, a_zeta, 0]
                    NN[i, a, 6] = Nxi[ik, a_xi, 1] * Neta[il, a_eta, 0] * Nzeta[im, a_zeta, 1]
                    NN[i, a, 7] = NN[i, a, 5]
                    NN[i, a, 8] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 2] * Nzeta[im, a_zeta, 0]
                    NN[i, a, 9] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 1] * Nzeta[im, a_zeta, 1]
                    NN[i, a, 10] = NN[i, a, 6]
                    NN[i, a, 11] = NN[i, a, 7]
                    NN[i, a, 12] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 0] * Nzeta[im, a_zeta, 2]

    return NN

def Lagrange_basis(degree, x, derivative=True):
    """Compute Lagrange shape function basis.

    Parameters
    ----------
    degree : int
        polynomial degree
    x : ndarray, 1D
        array containing the evaluation points of the polynomial
    derivative : bool
        whether to compute the derivative of the shape function or not
    returns : ndarray or (ndarray, ndarray)
        2D array of shape (len(x), degree + 1) containing the k = degree + 1 shape functions evaluated at x and optional the array containing the corresponding first derivatives 

    """
    if not hasattr(x, '__len__'):
        x = [x]
    nx = len(x)
    N = np.zeros((nx, degree + 1))
    for i, xi in enumerate(x):
        N[i] = __lagrange(xi, degree)
    if derivative == True:
        dN = np.zeros((nx, degree + 1))
        for i, xi in enumerate(x):
            dN[i] = __lagrange_x(xi, degree)
        return N, dN
    else:
        return N

def __lagrange(x, degree, skip=[]):
    """1D Lagrange shape functions, see https://en.wikipedia.org/wiki/Lagrange_polynomial#Definition.

    Parameter
    ---------
    x : float
        evaluation point
    degree : int
        polynomial degree
    returns : ndarray, 1D
        array containing the k = degree + 1 shape functions evaluated at x
    """
    k = degree + 1
    xi = np.linspace(-1, 1, num=k)
    l = np.ones(k)
    for j in range(k):
        for m in range(k):
            if m == j or m in skip:
                continue
            l[j] *= (x - xi[m]) / (xi[j] - xi[m])

    return l

def __lagrange_x(x, degree):
    """First derivative of 1D Lagrange shape functions, see https://en.wikipedia.org/wiki/Lagrange_polynomial#Derivatives.

    Parameter
    ---------
    x : float
        evaluation point
    degree : int
        polynomial degree
    returns : ndarray, 1D
        array containing the first derivative of the k = degree + 1 shape functions evaluated at x
    """
    k = degree + 1
    xi = np.linspace(-1, 1, num=k)
    l_x = np.zeros(k)
    for j in range(k):
        for i in range(k):
            if i == j:
                continue
            prod = 1
            for m in range(k):
                if m == i or m == j:
                    continue
                prod *= (x - xi[m]) / (xi[j] - xi[m])
            l_x[j] += prod / (xi[j] - xi[i])

    return l_x

def __lagrange_x_r(x, degree, skip=[]):
    """Recursive formular for first derivative of Lagrange shape functions.
    """
    k = degree + 1
    xi = np.linspace(-1, 1, num=k)
    l_x = np.zeros(k)
    for j in range(k):
        for i in range(k):
            if i == j or i in skip:
                continue
            l = __lagrange(x, degree, skip=[i] + skip)
            l_x[j] += l[j] / (xi[j] - xi[i])
            
    return l_x

def __lagrange_xx_r(x, degree):
    """Recursive formular for second derivative of Lagrange shape functions.
    """
    k = degree + 1
    xi = np.linspace(-1, 1, num=k)
    l_xx = np.zeros(k)
    for j in range(k):
        for i in range(k):
            if i == j:
                continue
            l_x = __lagrange_x_r(x, degree, skip=[i])
            l_xx[j] += l_x[j] / (xi[j] - xi[i])
            
    return l_xx

def test_shape_functions():
    degree = (1,2,3)
    xi = [0,1,2]
    eta = [3,4,5,6]
    zeta = [7,8,9,11]
    xis = (xi,eta,zeta)
    return lagrange_basis3D(degree,xis)

if __name__ == "__main__":
    import numpy as np

    #NN = test_shape_functions()

    degree = 2
    # x = -1
    # x = 0
    x = 1

    lx = __lagrange(x, degree)
    print(f'l({x}): {lx}')

    lx_x = __lagrange_x(x, degree)
    print(f'l_x({x}): {lx_x}')

    lx_x = __lagrange_x_r(x, degree)
    print(f'l_x({x}): {lx_x}')

    lx_xx = __lagrange_xx_r(x, degree)
    print(f'l_xx({x}): {lx_xx}')

    degree = 1
    derivative_order = 1
    x_array = np.array([-1, 0, 1])
    N, dN = Lagrange_basis(degree, x_array)
    print(f'N({x_array}):\n{N}')
    print(f'dN({x_array}):\n{dN}')