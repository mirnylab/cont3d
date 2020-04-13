import polychrom
import numpy as np 
import warnings
import h5py 
import glob
from polychrom.simulation import Simulation
import polychrom.starting_conformations
import polychrom.forces, polychrom.forcekits
from polychrom.polymer_analyses import Rg2_scaling, contact_scaling, kabsch_msd
import simtk.openmm 
import os 
import shutil
import polychrom.polymerutils
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file, save_hdf5_file
import matplotlib.pyplot as plt
import nglutils.nglutils as ngu
import nglview as nv
import pandas as pd
from cooltools.lib import numutils
from polychrom.contactmaps import monomerResolutionContactMap
from mirnylib.numutils import observedOverExpected
from random import randint
from sklearn.linear_model import LinearRegression
import pickle
from scipy.spatial import ckdtree
import mirnylib.plotting
from cooltools import numutils
from numba import jit
from multiprocessing import Pool
from functools import partial

@jit(nopython=True)
def triplets_from_contacts(N, contacts, dense=False, triplet_map=None):
    """
    A function to create triplets, in either list or dense form, from an array of 2D contacts.

    Parameters
    ----------

    N: int
        size of system
    contacts: numpy.ndarray 
        List of 2D contacts in the form [[u_1, v_1], [u_2, v_2], ...]
    dense: bool
        If True then return a dense array (a 3D numpy array)
        Defaults to False

    Returns
    -------
    A list of triplets (if dense is false)
    An N X N x N 3D contactmap (if dense is true)
    """
    num_contacts = contacts.shape[0]
    triplets = []
    
    # create adjacency matrix if one is not already given.
    adj_matrix = np.zeros((N, N))
    
    for i in range(num_contacts):
        adj_matrix[contacts[i][0]][contacts[i][1]] = 1
        
    i = 0 # start at first contact
    j = 1 # start at second contact
    
    while j < num_contacts:
        while contacts[i][0] == contacts[j][0]:
            if adj_matrix[contacts[i][1]][contacts[j][1]]:  # triplet found with second elements of i and j
                triplets.append([contacts[i][0], contacts[i][1], contacts[j][1]])

            j += 1 # advance second contact, but keep first contact where it is
        
        # j now has a different first element, we need to increment i and reset j back
        # i goes to next first element
        i += 1
        j = i + 1
    
    # just return a list of triplets if dense is False
    if not dense:
        raise NotImplementedError('triplet lists not implemented yet')
        # return np.array(triplets)

    # if a triplet map is given, we can just append to it
    # otherwise we need to create a new one (an expensive operation)
    if triplet_map is None:
        triplet_map = np.zeros((N, N, N))

    for triplet in triplets:
        triplet_map[triplet[0], triplet[1], triplet[2]] += 1

    return triplet_map

def _default_contact_finder(data, cutoff=5):
    """
    default finder of 2D contacts from an array of coordinates

    Parameters
    ----------
    data : Nx3 array
        Coordinates of points
    cutoff : float , optional
        Cutoff distance (contact radius)

    Returns
    -------
    k by 2 array of contacts. Each row corresponds to a contact.
    """
    conts = polychrom.polymer_analyses.calculate_contacts(data['pos'][:25000], cutoff=5)
    conts = np.unique(conts // 50, axis=0)
    return conts

def triplets_from_URI(
    N, 
    filename, 
    dense=False, 
    triplet_map=None, 
    contact_finder=polychrom.polymer_analyses.calculate_contacts, 
    cutoff=5
    ):
    """
    Calculates triplets from a polychrom conformation URI.

    Parameters
    ----------
    N: int
        size of system
    filename: string
        conformation URI
    triplet_map: string (optional)
        an N x N x N array to collect contacts (in dense form)
    dense: bool (optional)
        If True then return a dense array (a 3D numpy array)
        Defaults to False
    contact_finder: function (optional)
        A function that returns a list of unique contacts from a conformation. 
        E.g. polychrom.polymer_analyses.calculate_contacts


    Returns
    -------
    A list of triplets (if dense is false)
    An N X N x N 3D contactmap (if dense is true)
    """

    data = load_URI(filename)
    # conts = polychrom.polymer_analyses.calculate_contacts(data['pos'][:25000], cutoff=5)
    # conts = np.unique(conts // 50, axis=0)
    
    conts = contact_finder(data['pos'], cutoff=cutoff)
    triplets_from_contacts(N, conts, dense=dense, triplet_map=triplet_map)

    # note: the function adds the triplets to triplet_map and it also returns triplet_map
    return triplet_map
    

def triplets_from_bucket(
    N, 
    bucket, 
    contact_finder=polychrom.polymer_analyses.calculate_contacts
    ):
    """
    Calculates triplets from a bucket of URI's. This is the code that powers each thread.
    
    Parameters
    ---------

    N: int
        size of system
    bucket: iterable of string
        list of URI filenames where conformations are stored
    contact_finder: function (optional)
        A function that returns a list of unique contacts from a conformation. 
        E.g. polychrom.polymer_analyses.calculate_contacts

    
    """

    triplet_map = np.zeros((N, N, N))

    for filename in bucket:
        triplets_from_URI(N, filename, dense=True, triplet_map=triplet_map, contact_finder=contact_finder)

    return triplet_map

def triplet_map(
    N,
    URIs,
    n_threads=8,
    contact_finder=polychrom.polymer_analyses.calculate_contacts
):
    """
    Make a triplet map from a list of conformation URIs.

    Parameters
    ----------
    N: int
        size of system
    URIs: iterable of string
        list of URI filenames where conformations are stored
    contact_finder: function (optional)
        A function that returns a list of unique contacts from a conformation. 
        E.g. polychrom.polymer_analyses.calculate_contacts
    n_threads: int
        Number of threads to use. Defaults to 8.

    Returns
    -------
    A 3D N x N X N triplet-map
    """

    p = Pool(n_threads)
    URI_buckets = np.array_split(URIs, n_threads)

    mapped_arrays = p.map(partial(triplets_from_bucket, N, contact_finder=contact_finder), URI_buckets)    
    return np.sum(mapped_arrays, axis=0)
