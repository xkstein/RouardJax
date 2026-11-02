'''
I precomputed the eigenvalues of the rouard matrix in mathematica and created symbolic expressions

This code doesn't have dispersion and can only do gratings with one period and no tapers or anything, just handy because its very fast (fast enough to run on desmos actually)
'''

import numpy as np

def v1(d1, d2, r):
    '''U[2,1] * Inverse[U][1,2]'''
    return (np.exp(2j*d1)-np.exp(2j*(2*d1+d2))-np.exp(4j*d1)*r**2+np.exp(2j*(d1+d2))*r**2+np.sqrt(np.exp(4j*d1)*(1+np.exp(4j*(d1+d2))-2*np.exp(2j*d1)*r**2-2*np.exp(2j*d2)*r**2-2*np.exp(2j*(2*d1+d2))*r**2-2*np.exp(2j*(d1+2*d2))*r**2+np.exp(4j*d1)*r**4+np.exp(4j*d2)*r**4-2*np.exp(2j*(d1+d2))*(1-4*r**2+r**4))))/(2*np.sqrt(np.exp(4j*d1)*(1+np.exp(4j*(d1+d2))-2*np.exp(2j*d1)*r**2-2*np.exp(2j*d2)*r**2-2*np.exp(2j*(2*d1+d2))*r**2-2*np.exp(2j*(d1+2*d2))*r**2+np.exp(4j*d1)*r**4+np.exp(4j*d2)*r**4-2*np.exp(2j*(d1+d2))*(1-4*r**2+r**4))))

def v2(d1, d2, r):
    '''U[2,2] * Inverse[U][2,2]'''
    return (-np.exp(2j*d1)+np.exp(2j*(2*d1+d2))+np.exp(4j*d1)*r**2-np.exp(2j*(d1+d2))*r**2+np.sqrt(np.exp(4j*d1)*(1+np.exp(4j*(d1+d2))-2*np.exp(2j*d1)*r**2-2*np.exp(2j*d2)*r**2-2*np.exp(2j*(2*d1+d2))*r**2-2*np.exp(2j*(d1+2*d2))*r**2+np.exp(4j*d1)*r**4+np.exp(4j*d2)*r**4-2*np.exp(2j*(d1+d2))*(1-4*r**2+r**4))))/(2*np.sqrt(np.exp(4j*d1)*(1+np.exp(4j*(d1+d2))-2*np.exp(2j*d1)*r**2-2*np.exp(2j*d2)*r**2-2*np.exp(2j*(2*d1+d2))*r**2-2*np.exp(2j*(d1+2*d2))*r**2+np.exp(4j*d1)*r**4+np.exp(4j*d2)*r**4-2*np.exp(2j*(d1+d2))*(1-4*r**2+r**4))))

def E1(d1, d2, r):
    '''First eigenvalue'''
    return 0.5*np.exp(-1j*(3*d1+d2))*(np.exp(2j*d1)+np.exp(2j*(2*d1+d2))-np.exp(4j*d1)*r**2-np.exp(2j*(d1+d2))*r**2-np.sqrt(-4*np.exp(2j*(3*d1+d2))*(r**2-1)**2+np.exp(4j*d1)*(1+np.exp(2j*(d1+d2))-np.exp(2j*d1)*r**2-np.exp(2j*d2)*r**2)**2))

def E2(d1, d2, r):
    '''Second eigenvalue'''
    return 0.5*np.exp(-1j*(3*d1+d2))*(np.exp(2j*d1)+np.exp(2j*(2*d1+d2))-np.exp(4j*d1)*r**2-np.exp(2j*(d1+d2))*r**2+np.sqrt(-4*np.exp(2j*(3*d1+d2))*(r**2-1)**2+np.exp(4j*d1)*(1+np.exp(2j*(d1+d2))-np.exp(2j*d1)*r**2-np.exp(2j*d2)*r**2)**2))

def analytical_grating_transmission(wavelength, n_high, n_low, length_high, length_low, N):
    '''
    Find transmission of simple high/low gratings

    Args:
        wavelength: array of wavelengths
        n_high: high film index
        n_low: low film index
        length_high: length of high index film
        length_low: length of low index film
        N: number of periods in grating

    Returns:
        |T|^2
    '''
    r = ( n_high - n_low ) / ( n_high + n_low )
    t = np.sqrt( 1 - r ** 2 )
    d1 = 2 * np.pi * n_high / wavelength * length_high
    d2 = 2 * np.pi * n_low / wavelength * length_low
    return np.abs(t) ** ( 2 * N ) / np.abs( E1(d1, d2, r) ** N * v1(d1, d2, r) + E2(d1, d2, r) ** N * v2(d1, d2, r) ) ** 2

if __name__ == '__main__':
    assert np.isclose(v1(1,1,1), 0.5+0.32104631j)
    assert np.isclose(v2(1,1,1), 0.5-0.32104631j)
    assert np.isclose(E1(1,1,1), 7.2569e-17-8.4022e-17j)
    assert np.isclose(E2(1,1,1), -2.83229367)
    
    assert np.isclose(v1(2,1,100), 1.00011681+1.63e-16j)
    assert np.isclose(v2(2,1,100), -0.00011681-1.23e-16j)
    assert np.isclose(E1(2,1,100), -5404.01305+8412.88559j)
    assert np.isclose(E2(2,1,100), -5404.01305-8412.88559j)