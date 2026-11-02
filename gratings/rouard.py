'''
This simulates bragg gratings using rouard's method, and it uses jax to 
parrallelize computation across wavelengths

Reference: Chapter 4 - Theory of Fiber Bragg Gratings by Raman Kashyap
'''
import jax 
import jax.numpy as jnp
import numpy as np

vectorized_multi_dot = jax.jit(jax.vmap(jax.numpy.linalg.multi_dot))

def rouard_jnp(wavelengths, layer_length, layer_neff):
    '''Applies Rouard's method in parallel

    wavelengths (1xN): Wavelengths to evaluate grating structure at
    layer_lengths (1xM): Length of layers in grating structure
    layer_neff (NxM): neff for each layer at each wavelength (layer_neff[0] should be neff of all the layers at wavelengths[0])
    '''
    r = ( layer_neff[:,1:] - layer_neff[:,:-1] ) / ( layer_neff[:,1:] + layer_neff[:,:-1] )
    phi = -2j * jnp.pi * jnp.einsum('ij,j,i->ij', layer_neff, layer_length, 1 / wavelengths)[:,:-1]

    layers = jnp.array([[ jnp.exp(phi),  r * jnp.exp(phi) ], [ r * jnp.exp(-phi), jnp.exp(-phi) ]])
    layers = jnp.moveaxis(layers, [-2, -1], [0, 1])

    S = vectorized_multi_dot(layers)
    return S

def rouard(wavelengths, layer_length, layer_neff, alpha=0):
    '''Applies Rouard's method in parallel

    The last layer length is neglected
    
    wavelengths (1xN): Wavelengths to evaluate grating structure at
    layer_lengths (1xM): Length of layers in grating structure
    layer_neff (NxM): neff for each layer at each wavelength (layer_neff[0] should be neff of all the layers at wavelengths[0])
    '''
    r = ( layer_neff[:,1:] - layer_neff[:,:-1] ) / ( layer_neff[:,1:] + layer_neff[:,:-1] )
    phi = -2j * np.pi * np.einsum('ij,j,i->ij', layer_neff[:,:-1], layer_length[:-1], 1 / wavelengths)
    # print(phi.shape, len(layer_length))
    phi -= alpha * np.array(len(wavelengths) * [layer_length[:-1]])

    layers = np.array([[ np.exp(phi),  r * np.exp(phi) ], [ r * np.exp(-phi), np.exp(-phi) ]])
    layers = np.moveaxis(layers, [-2, -1], [0, 1])

    S = vectorized_multi_dot(layers)
    return S
