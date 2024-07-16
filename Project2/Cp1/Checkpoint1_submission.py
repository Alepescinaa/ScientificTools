import numpy as np
import porepy as pp
import scipy.sparse as sps
import scipy

import sys


# Depending on the setting, we then need to setup the local path for importing some useful functions

main_folder = "./"
spe10_folder = main_folder + "spe10"
sys.path.insert(1, spe10_folder)

from functions import *
from spe10 import Spe10


# Define the function that given a subdomain, its data, and a direction compute the upscaled gradient and flux.


def upscale(sd, perm, dir, export_folder=None):
    """
    Compute the averaged gradient and flux for a given subdomain and direction of the pressure
    gradient.

    Args:
        sd (pp.Grid): The grid representing the subdomain.
        perm (dict): The permeability of the subdomain divided in the fields "kxx" and "kyy"
        dir (int): The direction of the flow, 0 means x-direction and 1 means y-direction.
        export_folder (str): If given, path where to export the results.
            Default to None, no exporting.

    Returns:
        (np.ndarray, np.ndarray): averaged gradient and flux.
    """
    # Permeability
    perm_tensor = pp.SecondOrderTensor(kxx=perm["kxx"], kyy=perm["kyy"])

    # Boundary conditions
    b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = sd.face_centers[:, b_faces]

    # Find the min and max values of the boundary faces
    sd_min = np.amin(sd.face_centers[dir, :])
    sd_max = np.amax(sd.face_centers[dir, :])

    # define outflow and inflow type boundary conditions
    out_flow = np.isclose(b_face_centers[dir, :], sd_max)
    in_flow = np.isclose(b_face_centers[dir, :], sd_min)

    # define the labels and values for the boundary faces
    labels = np.array(["neu"] * b_faces.size)
    labels[np.logical_or(in_flow, out_flow)] = "dir"

    bc_val = np.zeros(sd.num_faces)
    bc_val[b_faces[in_flow]] = sd_max - sd_min

    bc = pp.BoundaryCondition(sd, b_faces, labels)

    # Collect all parameters in a dictionary
    key = "flow"
    parameters = {"second_order_tensor": perm_tensor, "bc": bc, "bc_values": bc_val}
    data = pp.initialize_default_data(sd, {}, key, parameters)

    # Discretize the problem (construct the lhr and rhs)
    #discr = TODO
    #discr.discretize TODO
    discr = pp.Tpfa(key)
    discr.discretize(sd, data)

    #A, b = TODO
    A, b = discr.assemble_matrix_rhs(sd, data)

    # Solve the linear system and compute the pressure
    # p = TODO
    p = sps.linalg.spsolve(A, b)

    # Export the solution
    if export_folder is not None:
        save = pp.Exporter(sd, "sol", folder_name=export_folder)
        save.write_vtu([("p", p), ("log_perm", np.log10(perm["kxx"]))])

    # Post-process the solution to get the flux
    return compute_avg_q_grad(sd, p, data, key, bc, bc_val)


# Define the function that compute the symmetric upscaled tensor.


def compute_tensor(grad_h, grad_v, q_h, q_v):
    """
    Compute the upscaled permeability tensor.

    Args:
        grad_h (np.ndarray): Gradient in the horizontal direction.
        grad_v (np.ndarray): Gradient in the vertical direction.
        q_h (np.ndarray): Flux in the horizontal direction.
        q_v (np.ndarray): Flux in the vertical direction.

    Returns:
        perm (np.ndarray): Upscaled permeability tensor.

    The function solves a linear system to obtain the upscaled permeability tensor
    based on the given gradients and fluxes. It enforces numerical symmetry and
    checks if the resulting tensor is symmetric positive definite (SPD).
    """
    # Solve the linear system to get the upscaled permeability
    # TODO
    lhs = np.array([
        [grad_h[0], grad_h[1], 0, 0],
        [0, 0, grad_h[0], grad_h[1]],
        [grad_v[0], grad_v[1], 0, 0],
        [0, 0, grad_v[0], grad_v[1]],
        [0, 1, -1, 0]
    ])

    rhs = np.array([q_h[0], q_h[1], q_v[0], q_v[1], 0])

    perm = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    # make it symmetric positive definite
    perm = nearest_spd(perm.reshape(2, 2)).ravel()

    return perm


# Perform the upscaling for the Spe10 benchmark of a given layer.


import time
selected_layers = 20
folder_results = main_folder + "results/"

# start = time.time()
# #sd_coarse, result = Checkpoint1_solution(selected_layers, folder_results)
# end = time.time()
# print(end - start)


from pathos.multiprocessing import _ProcessPool as Pool

#print(multiprocessing.cpu_count())

def process_subdomain(sub_sd_id, sub_sd, perm_dict, folder_results, part, kxx_up, kxy_up, kyx_up, kyy_up):
    mask = part == sub_sd_id
    sub_perm = {key: val[mask] for key, val in perm_dict.items()}

    folder_x = folder_results + str(sub_sd_id) + "_x"
    q_h, grad_h = upscale(sub_sd, sub_perm, 0, folder_x)

    folder_y = folder_results + str(sub_sd_id) + "_y"
    q_v, grad_v = upscale(sub_sd, sub_perm, 1, folder_y)

    kk = compute_tensor(grad_h, grad_v, q_h, q_v)

    return [kk[0], kk[1], kk[2], kk[3]]

def Checkpoint1_solution(selected_layers, folder_results):
    spe10 = Spe10(selected_layers)
    perm_folder = spe10_folder + "/perm/"
    spe10.read_perm(perm_folder)
    perm_dict = spe10.perm_as_dict()

    num_part = 20
    part, sub_sds, sd_coarse = coarse_grid(spe10.sd, num_part)

    kxx_up = np.zeros(spe10.sd.num_cells)
    kxy_up = np.zeros(spe10.sd.num_cells)
    kyx_up = np.zeros(spe10.sd.num_cells)
    kyy_up = np.zeros(spe10.sd.num_cells)
    kxx = np.zeros(spe10.sd.num_cells)

    result = []
    args = [(sub_sd_id, sub_sd, perm_dict, part) for sub_sd_id, sub_sd in enumerate(sub_sds)]
    with Pool() as pool:
        for kk, mask in pool.starmap(process_subdomain, args, chunksize = 2):
            kxx_up[mask], kxy_up[mask], kyx_up[mask], kyy_up[mask] = kk
            result.append(kk)
            
    var_to_save = [
        ("kxx", kxx_up),
        ("kxy", kxy_up),
        ("kyx", kyx_up),
        ("kyy", kyy_up),
        ("fine", kxx)
    ]

    save = pp.Exporter(spe10.sd, "upscaled_k", folder_name=folder_results)
    save.write_vtu(var_to_save)

    write_upscaled_perm("as_tensor", sd_coarse, result, folder_results)

    return sd_coarse, result

# import time

if __name__ == '__main__':

    selected_layers = 20
    folder_results = main_folder + "results/"

    start = time.time()
    sd_coarse, result = Checkpoint1_solution(selected_layers, folder_results)
    end = time.time()
    print(end - start)


