import polychrom
import numpy as np 
from polychrom.hdf5_format import load_URI
from scipy.spatial import ckdtree
from numba import jit
from multiprocessing import Pool
from functools import partial
import polychrom.polymer_analyses

@jit(nopython=True)
def dense_triplets_from_contacts(N, contacts, triplet_map=None):
    """
    A function to create triplets, in dense form, from an array of 2D contacts.

    Parameters
    ----------

    N: int
        size of system
    contacts: numpy.ndarray 
        List of 2D contacts in the form [[u_1, v_1], [u_2, v_2], ...]
    triplet_map: numpy.ndarray 
        If specified, contacts will be added to this matrix. A new matrix will not be created.
        Use this if averaging over a lot of conformations.

    Returns
    -------
    An N X N x N 3D contactmap, where map[u][v][w] == 1 corresponds to a triplet between u, v, and w.
    """
    num_contacts = contacts.shape[0]
    triplets = []
    
    # create adjacency matrix if one is not already given.
    adj_matrix = np.zeros((N, N))
    
    for i in range(num_contacts):
        adj_matrix[contacts[i][0]][contacts[i][1]] = 1
        
    i = 0 # start at first contact
    j = 0 # start at first contact
    
    while j < num_contacts:
        while contacts[i][0] == contacts[j][0]:
            if adj_matrix[contacts[i][1]][contacts[j][1]]:  # triplet found with second elements of i and j
                triplets.append([contacts[i][0], contacts[i][1], contacts[j][1]])

            j += 1 # advance second contact, but keep first contact where it is
        
        # j now has a different first element, we need to increment i and reset j back
        # i goes to next first element
        i += 1
        j = i
    
    # if a triplet map is given, we can just append to it
    # otherwise we need to create a new one (an expensive operation)
    if triplet_map is None:
        triplet_map = np.zeros((N, N, N))

    for triplet in triplets:
        triplet_map[triplet[0], triplet[1], triplet[2]] += 1

    return triplet_map


@jit(nopython=True)
def sparse_triplets_from_contacts(N, contacts):
    """
    A function to create triplets in list form from an array of 2D contacts. 
    Works similarly to dense_triplets_from_contacts.

    Parameters
    ----------

    N: int
        size of system
    contacts: numpy.ndarray 
        List of 2D contacts in the form [[u_1, v_1], [u_2, v_2], ...]

    Returns
    -------
    A list of triplets of the form [[u_1, v_1, w_1], [u_2, v_2, w_2], ...]
    where [u_i, v_i, w_i] is a triplet.
    """
    num_contacts = contacts.shape[0]
    triplets = []
    
    # create adjacency matrix if one is not already given.
    adj_matrix = np.zeros((N, N))
    
    for i in range(num_contacts):
        adj_matrix[contacts[i][0]][contacts[i][1]] = 1
        
    i = 0 # start at first contact
    j = 0 # start at first contact
    
    while j < num_contacts:
        while contacts[i][0] == contacts[j][0]:
            if adj_matrix[contacts[i][1]][contacts[j][1]]:  # triplet found with second elements of i and j
                triplets.append([contacts[i][0], contacts[i][1], contacts[j][1]])

            j += 1 # advance second contact, but keep first contact where it is
        
        # j now has a different first element, we need to increment i and reset j back
        # i goes to next first element
        i += 1
        j = i
    
    return np.array(triplets)


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
    cutoff: float
        Radius cutoff for contactmaps


    Returns
    -------
    A list of triplets (if dense is false)
    An N X N x N 3D contactmap (if dense is true)
    """

    data = load_URI(filename)
    # conts = polychrom.polymer_analyses.calculate_contacts(data['pos'][:25000], cutoff=5)
    # conts = np.unique(conts // 50, axis=0)
    
    conts = contact_finder(data['pos'], cutoff=cutoff)
    if dense:
        dense_triplets_from_contacts(N, conts, triplet_map=triplet_map)

        # note: the function adds the triplets to triplet_map and it also returns triplet_map
        return triplet_map

    return sparse_triplets_from_contacts(N, conts)


def triplets_from_bucket(
    N, 
    bucket, 
    contact_finder=polychrom.polymer_analyses.calculate_contacts,
    cutoff=5
    ):
    """
    Calculates triplets from a bucket of URI's. This is the code that powers each thread.
    Note: only dense triplets are returned by this function
    
    Parameters
    ---------

    N: int
        size of system
    bucket: iterable of string
        list of URI filenames where conformations are stored
    contact_finder: function (optional)
        A function that returns a list of unique contacts from a conformation. 
        contact_finder should be specified to keep the size of the map under control. 
        Coursegraining or trimming the coordinate matrix should be included in contact finder.
        E.g. polychrom.polymer_analyses.calculate_contacts (default)
    cutoff: float
        Radius cutoff for contactmaps

    Returns
    -------
    An N X N x N 3D contactmap, where map[u][v][w] is the total number of contacts between u, v, and w.
    """

    triplet_map = np.zeros((N, N, N))

    for filename in bucket:
        triplets_from_URI(N, filename, dense=True, triplet_map=triplet_map, contact_finder=contact_finder, cutoff=cutoff)

    return triplet_map

def triplet_map(
    N,
    URIs,
    n_threads=8,
    contact_finder=polychrom.polymer_analyses.calculate_contacts,
    cutoff=5
):
    """
    Make a triplet map from a list of conformation URIs.
    A multi-threaded version of triplets_from_bucket.

    Parameters
    ----------
    N: int
        size of system
    URIs: iterable of string
        list of URI filenames where conformations are stored
    contact_finder: function (optional)
        A function that returns a list of unique contacts from a conformation. 
        contact_finder should be specified to keep the size of the map under control. 
        Coursegraining or trimming the coordinate matrix should be included in contact finder.
        E.g. polychrom.polymer_analyses.calculate_contacts (default)
    n_threads: int
        Number of threads to use. Defaults to 8.
    cutoff: float
        Radius cutoff for contactmaps

    Returns
    -------
    An N X N x N 3D contactmap, where map[u][v][w] is the total number of contacts between u, v, and w.
    """

    URI_buckets = np.array_split(URIs, n_threads)

    with Pool(n_threads) as p:    
        mapped_arrays = p.map(partial(triplets_from_bucket, N, contact_finder=contact_finder, cutoff=cutoff), URI_buckets)    
    
    return np.sum(mapped_arrays, axis=0)
