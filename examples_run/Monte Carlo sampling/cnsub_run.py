# Modify by Zhong-Lun Li Sep. 16 2025,
# Department of Chemical Engineering, National Taiwan University of Science and Technology, Taipei 106, Taiwan

# The CHGNet model comes from this study:
# Deng, B., Zhong, P., Jun, K. et al. CHGNet as a pretrained universal neural network potential for
# charge-informed atomistic modelling. Nat Mach Intell 5, 1031–1041 (2023). https://doi.org/10.1038/s42256-023-00716-3

import os, time, io, sys, warnings, torch, shutil, hashlib, json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from multiprocessing import Process

from ase.gui.surfaceslab import structures
from chgnet.model.model import CHGNet
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element


def rd_pos(path_f):
    with open(path_f, 'r') as nf_rf:
        d = nf_rf.read()
        d = d.split("\n")
    lc = np.zeros((3, 3))
    atom_type = np.zeros(0)
    atom_num = np.zeros(0)
    for nif2 in range(3):
        lc[nif2, :] = np.array([float(d[2 + nif2].split()[0]),
                                float(d[2 + nif2].split()[1]),
                                float(d[2 + nif2].split()[2])])
    lc *= float(d[1].split()[0])

    for nif2 in range(len(d[5].split())):
        atom_type = np.concatenate((atom_type, np.array([d[5].split()[nif2]])))
        atom_num = np.concatenate((atom_num, np.array([int(d[6].split()[nif2])])))
    atom_num_sum = int(np.sum(atom_num))

    if ("S" in d[7]) or ("s" in d[7]):  # Selective dynamics
        st = 9
    else:
        st = 8

    coord = np.zeros((atom_num_sum, 3))
    coord_relax = np.zeros((atom_num_sum, 3)) + 1
    for nif in range(atom_num_sum):
        coord[nif, :] = np.array([float(d[st + nif].split()[0]),
                                  float(d[st + nif].split()[1]),
                                  float(d[st + nif].split()[2])])
        for nif2 in range(len(d[st + nif].split()) - 3):
            if d[st + nif].split()[3 + nif2] == "F":
                coord_relax[nif, nif2] = 0

    if ("D" in d[st - 1]) or ("d" in d[st - 1]):
        coord = np.dot(coord, lc)

    if st == 8:
        coord_relax[:, :] = 1

    from ase import io
    from pymatgen.io.ase import AseAtomsAdaptor
    atoms_f = io.read(path_f)
    atoms_f.pbc = [PBC_A, PBC_B, PBC_C]

    return d[0], atom_type, atom_num, lc, coord, coord_relax, AseAtomsAdaptor.get_structure(atoms_f)


def rd_cncar(path_f):
    with open(path_f, 'r') as nf_f:
        data_f = nf_f.read()
        data_f = data_f.split("\n")
    for nif in range(len(data_f)):
        data_nif = data_f[nif]
        if "#" in data_f[nif]:
            data_nif = data_nif[:data_nif.index("#")]
        data_nif = data_nif.split()
        if len(data_nif) == 0:
            continue

        if data_nif[0] == "IBRION":
            global IBRION
            IBRION = int(data_nif[-1])
        elif data_nif[0] == "PBC_A":
            global PBC_A
            if int(data_nif[-1]) != 1:
                PBC_A = False
        elif data_nif[0] == "PBC_B":
            global PBC_B
            if int(data_nif[-1]) != 1:
                PBC_B = False
        elif data_nif[0] == "PBC_C":
            global PBC_C
            if int(data_nif[-1]) != 1:
                PBC_C = False
        elif data_nif[0] == "EDIFFG":
            global EDIFFG
            EDIFFG = abs(float(data_nif[-1]))
        elif data_nif[0] == "NSW_OPT":
            global NSW_OPT
            NSW_OPT = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "OFI_OPT":
            global OFI_OPT
            OFI_OPT = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "ISIF":
            global ISIF
            if int(data_nif[-1]) == 2:
                ISIF = False
            elif int(data_nif[-1]) == 3:
                ISIF = True
        elif data_nif[0] == "POTIM_PH":
            global POTIM_PH
            POTIM_PH = float(data_nif[-1])
        elif data_nif[0] == "CUT_WN":
            global CUT_WN
            CUT_WN = float(data_nif[-1])
        elif data_nif[0] == "TEMP_STA":
            global TEMP_STA
            TEMP_STA = float(data_nif[-1])
        elif data_nif[0] == "GAU_FWHM":
            global GAU_FWHM
            GAU_FWHM = float(data_nif[-1])
        elif data_nif[0] == "POTIM":
            global POTIM
            POTIM = float(data_nif[-1])
        elif data_nif[0] == "NSW_MD":
            global NSW_MD
            NSW_MD = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "OFI_MD":
            global OFI_MD
            OFI_MD = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "ENSB":
            global ENSB
            if int(data_nif[-1]) == 1:  # NVE
                ENSB = "nve"
            elif int(data_nif[-1]) == 2:  # NVT
                ENSB = "nvt"
            elif int(data_nif[-1]) == 3:  # NPT
                ENSB = "npt"
        elif data_nif[0] == "THEMT":
            global THEMT
            if int(data_nif[-1]) == 0:  # Nose-Hoover
                THEMT = "Nose-Hoover"
            elif int(data_nif[-1]) == 1:  # Berendsen
                THEMT = "Berendsen"
            elif int(data_nif[-1]) == 2:  # Berendsen_inhomogeneous
                THEMT = "Berendsen_inhomogeneous"
        elif data_nif[0] == "TEMP_B":
            global TEMP_B
            TEMP_B = float(data_nif[-1])
        elif data_nif[0] == "TEMP_E":
            global TEMP_E
            TEMP_E = float(data_nif[-1])
        elif data_nif[0] == "TAU_T":
            global TAU_T
            TAU_T = max([1, float(data_nif[-1])])
        elif data_nif[0] == "TAU_P":
            global TAU_P
            TAU_P = max([1, float(data_nif[-1])])
        elif data_nif[0] == "PRES":
            global PRES
            PRES = float(data_nif[-1]) / 1E4
        elif data_nif[0] == "BULK_M":
            global BULK_M
            BULK_M = float(data_nif[-1])
        elif data_nif[0] == "FASEQ":
            global FASEQ
            FASEQ = data_nif[2:]
        elif data_nif[0] == "NAME_OCC":
            global NAME_OCC
            NAME_OCC = data_nif[2:]
        elif data_nif[0] == "NUM_OCC":
            global NUM_OCC
            NUM_OCC = data_nif[2:]
        elif data_nif[0] == "NSW_MC_G":
            global NSW_MC_G
            NSW_MC_G = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "NSW_MC_S":
            global NSW_MC_S
            NSW_MC_S = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "EDIFFG0":
            global EDIFFG0
            EDIFFG0 = float(data_nif[-1])
        elif data_nif[0] == "TEMP_MC":
            global TEMP_MC
            TEMP_MC = float(data_nif[-1])
        elif data_nif[0] == "SCQU1":
            global SCQU1
            SCQU1 = int(float(data_nif[-1]))
        elif data_nif[0] == "EDIFFG1":
            global EDIFFG1
            EDIFFG1 = abs(float(data_nif[-1]))
        elif data_nif[0] == "NSW_OPT1":
            global NSW_OPT1
            NSW_OPT1 = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "OFI_OPT1":
            global OFI_OPT1
            OFI_OPT1 = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "ISIF1":
            global ISIF1
            if int(data_nif[-1]) == 2:
                ISIF1 = False
            elif int(data_nif[-1]) == 3:
                ISIF1 = True
        elif data_nif[0] == "SCQU2":
            global SCQU2
            SCQU2 = int(float(data_nif[-1]))
        elif data_nif[0] == "EDIFFG2":
            global EDIFFG2
            EDIFFG2 = abs(float(data_nif[-1]))
        elif data_nif[0] == "NSW_OPT2":
            global NSW_OPT2
            NSW_OPT2 = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "OFI_OPT2":
            global OFI_OPT2
            OFI_OPT2 = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "ISIF2":
            global ISIF2
            if int(data_nif[-1]) == 2:
                ISIF2 = False
            elif int(data_nif[-1]) == 3:
                ISIF2 = True
        elif data_nif[0] == "NSW_MCCE":
            global NSW_MCCE
            NSW_MCCE = int(float(data_nif[-1]))
        elif data_nif[0] == "E_MCCE":
            global E_MCCE
            E_MCCE = float(data_nif[-1])
        elif data_nif[0] == "EDIFFG3":
            global EDIFFG3
            EDIFFG3 = abs(float(data_nif[-1]))
        elif data_nif[0] == "NSW_OPT3":
            global NSW_OPT3
            NSW_OPT3 = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "OFI_OPT3":
            global OFI_OPT3
            OFI_OPT3 = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "ISIF3":
            global ISIF3
            if int(data_nif[-1]) == 2:
                ISIF3 = False
            elif int(data_nif[-1]) == 3:
                ISIF3 = True
        elif data_nif[0] == "POTIM_CE":
            global POTIM_CE
            POTIM_CE = float(data_nif[-1])
        elif data_nif[0] == "CUT_WNCE":
            global CUT_WNCE
            CUT_WNCE = float(data_nif[-1])
        elif data_nif[0] == "TEMP_CE":
            global TEMP_CE
            TEMP_CE = float(data_nif[-1])
        elif data_nif[0] == "CVG_S_CE":
            global CVG_S_CE
            CVG_S_CE = float(data_nif[-1])


def gen(path_f, st_inp, tp, print_time, chg_line):  # Generate a text string and log it to the output file
    if print_time:
        st_new = "[" + str(datetime.now().year) + "-" + "%02d" % datetime.now().month + "-" + \
                 "%02d" % datetime.now().day + "] [" + "%02d" % datetime.now().hour + ":" + \
                 "%02d" % datetime.now().minute + ":" + "%02d" % datetime.now().second + "]  " + st_inp
        if chg_line:
            if tp:
                print(st_new)
            with open(path_f, 'a') as wr_ff:
                wr_ff.write(st_new + "\n")
        else:
            if tp:
                print(st_new, end="")
            with open(path_f, 'a') as wr_ff:
                wr_ff.write(st_new)
    else:
        if chg_line:
            if tp:
                print(st_inp)
            with open(path_f, 'a') as wr_ff:
                wr_ff.write(st_inp + "\n")
        else:
            if tp:
                print(st_inp, end="")
            with open(path_f, 'a') as wr_ff:
                wr_ff.write(st_inp)


def wr_cncar():
    gen(Path_log, '"Input_CHGNet" does not exist. The program will automatically generate it.', True, True, True)
    gen(Path_log, "Please change the settings and resubmit the job.", True, True, True)
    with open(os.path.join(Path_job, "Input_CHGNet"), 'w') as nf_rf:
        nf_rf.write("# " + st_str + "\n\n"
                    "# General\n"
                    "#  Tags      Options   Default  Comments\n"
                    "  IBRION   =  -1       # [-1]\n"
                    "  # single point (-1); molecular dynamic (0); optimize using RMM-DIIS (1) or Damped MD (3); \n"
                    "  # phonon calculation (5); optimize with phonon calculation "
                                    "using RMM-DIIS (15) or Damped MD (35);\n"
                    "  # sampling optimization using RMM-DIIS (11) or Damped MD (13)\n"
                    "  PBC_A    =   1       # [1]    # considering the periodicity in the a vector (1) or not (0)\n"
                    "  PBC_B    =   1       # [1]    # considering the periodicity in the b vector (1) or not (0)\n"
                    "  PBC_C    =   1       # [1]    # considering the periodicity in the c vector (1) or not (0)\n"
                    "\n\n# For optimization\n"
                    "#  Tags      Options   Default  Comments\n"
                    "  EDIFFG   =  1E-2     # [0.01] # convergence condition, maximum force on atoms (in eV/Angstrom)\n"
                    "  NSW_OPT  =  500      # [500]  # maximum number of steps for geometry optimization\n"
                    "  OFI_OPT  =   1       # [1]    # output frame interval\n"
                    "  ISIF     =   3       # [3]    # fix cell (2); relax cell (3)\n"
                    "\n\n# For phonon calculation (not related to Monte Carlo calculations here)\n"
                    "#  Tags      Options   Default  Comments\n"
                    "  POTIM_PH =  0.01     # [0.01] # the displacement of each atom (in Angstrom)\n"
                    "  CUT_WN   =   50      # [50]   # wave number calculation lower limit (in cm^-1)\n"
                    "  TEMP_STA =  298      # [298]  # temperature of statistical thermodynamics (in K)\n"
                    "  GAU_FWHM =   10      # [10]   # full width at half maximum (FWHM) "
                                    "of the Gaussian smoothing (in cm-1)\n"
                    "\n\n# For MD\n"
                    "#  Tags      Options   Default  Comments\n"
                    "  POTIM    =   1       # [1]    # time step for MD (in fs)\n"
                    "  NSW_MD   =  100      # [100]  # the number of time step for MD\n"
                    "  OFI_MD   =   1       # [1]    # output frame interval\n"
                    "  ENSB     =   2       # [2]    # ensemble selection, NVE (1); NVT (2); NPT (3)\n"
                    "  THEMT    =   2       # [2]    # thermostat selection, Nose-Hoover (0); Berendsen (1); "
                    "Berendsen_inhomogeneous (2)\n"
                    "  TEMP_B   =  298      # [298]  # starting temperature in K\n"
                    "  TEMP_E   =  298      # [298]  # end temperature in K\n"
                    "  TAU_T    =  100      # [100]  # time constant for temperature coupling in fs\n"
                    "  TAU_P    = 1000      # [1000] # time constant for pressure coupling in fs (NPT)\n"
                    "  PRES     =   1       # [1]    # pressure in bar (NPT)\n"
                    "  BULK_M   =   2       # [2]    # bulk modulus of the material in GPa (NPT)\n"
                    "\n\n# For sampling optimization\n"
                    "#  Tags      Options   Default  Comments\n"
                    "# FASEQ    = \n"
                    '  # FASEQ: the serial numbers start from 0. '
                    'Different serial numbers are separated by space and written in the same column. \n'
                    "  #                                         "
                                    "You can use colons to represent consecutive serial numbers.\n"
                    "  #                                         "
                                    "The following two examples show the same representation method, \n"
                    "  #                                         0,1,3,5,10, 11, 12, 18, 21\n"
                    "  #                                         0, 1:2:5, 10:12, 18, 21\n"
                    "# NAME_OCC = \n"
                    "# NUM_OCC = \n"
                    '  # NAME_OCC: the name of the atom to be placed, if it is vacancy, the name is "Va"\n'
                    "  # NUM_OCC: the number of atoms to be placed\n"
                    "  NSW_MC_G =   1       # [1]    # number of Monte Carlo iteration groups\n"
                    "  NSW_MC_S =  1E3      # [1E3]  # number of Monte Carlo iterations in the 1st stage of screening\n"
                    "  EDIFFG0  =  1E-5     # [1E-5] # convergence condition of Monte Carlo method (in eV/atoms)\n"
                    "  TEMP_MC  =  298      # [298]  # temperature of Monte Carlo method (in K)\n\n"
                    "# 1st stage\n"
                    "  SCQU1    =   10      # [10]   # screening quantity\n"
                    "  EDIFFG1  =  5E-2     # [0.05] # convergence condition, maximum force on atoms (in eV/Angstrom)\n"
                    "  NSW_OPT1 =   50      # [50]   # maximum number of steps for geometry optimization\n"
                    "  OFI_OPT1 =   1       # [1]    # output frame interval\n"
                    "  ISIF1    =   3       # [3]    # fix cell (2); relax cell (3)\n\n"
                    "# 2nd stage\n"
                    "  SCQU2    =   3       # [3]    # screening quantity for all sampling data\n"
                    "  EDIFFG2  =  1E-2     # [0.01] # convergence condition, maximum force on atoms (in eV/Angstrom)\n"
                    "  NSW_OPT2 =  500      # [500]  # maximum number of steps for geometry optimization\n"
                    "  OFI_OPT2 =   1       # [1]    # output frame interval\n"
                    "  ISIF2    =   3       # [3]    # fix cell (2); relax cell (3)\n\n"
                    "# 3rd stage (configuration entropy, the settings for phonon calculations are irrelevant here)\n"
                    "  NSW_MCCE =   0       # [0]    # number of Monte Carlo iterations "
                                    "in the configuration entropy calculation\n"
                    "  E_MCCE   =   3       # [3]    "
                                    "# The upper limit of the energy sampling of the vibration entropy, \n"
                    "                                "
                                    "# only structures below this energy will be included in the calculation (in eV)\n"
                    "  EDIFFG3  =  1E-2     # [0.01] # convergence condition, maximum force on atoms (in eV/Angstrom)\n"
                    "  NSW_OPT3 =  100      # [100]  # maximum number of steps for geometry optimization\n"
                    "  OFI_OPT3 =   1       # [1]    # output frame interval\n"
                    "  ISIF3    =   3       # [3]    # fix cell (2); relax cell (3)\n"
                    "  POTIM_CE =  0.01     # [0.01] # the displacement of each atom (in Angstrom)\n"
                    "  CUT_WNCE =   50      # [50]   # wave number calculation lower limit (in cm^-1)\n"
                    "  TEMP_CE  =  298      # [298]  # temperature of statistical thermodynamics (in K)\n"
                    "  CVG_S_CE =  1E-6     # [1E-6] # convergence conditions of "
                                    "the configuration entropy slope (in meV/atoms/K/steps)\n\n")

    exit(0)


def num_printer(data, sp, dg, tp):
    # tp: full number: 0, scientific notation: 1
    if abs(data) <= 10 ** -(dg + 3):
        data = 0
    blank_str = ""
    if np.abs(data) < 1E-50:
        if tp == 1:
            for _ in range(int(sp - dg - 6)):
                blank_str += " "
            return blank_str + format(data, '.' + str(dg) + 'E')
        else:
            if 1 > sp:
                return format(data, '.' + str(dg) + 'f')
            else:
                for _ in range(int(sp - dg - 2)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'f')
    if data > 0:  # positive
        if tp == 1:
            if data <= 1E-100:
                for _ in range(int(sp - dg - 7)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'E')
            elif data >= 1E100:
                for _ in range(int(sp - dg - 7)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'E')
            else:
                for _ in range(int(sp - dg - 6)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'E')
        else:
            od = max(1, np.floor(np.log10(data)) + 1)
            if od > sp:
                return format(data, '.' + str(dg) + 'f')
            else:
                for _ in range(int(sp - od - dg - 1)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'f')
    else:  # negative
        if tp == 1:
            if data >= -1E-100:
                for _ in range(int(sp - dg - 8)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'E')
            elif data <= -1E100:
                for _ in range(int(sp - dg - 8)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'E')
            else:
                for _ in range(int(sp - dg - 7)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'E')
        else:
            od = max(1, np.floor(np.log10(-data)) + 1)
            if od > sp:
                return format(data, '.' + str(dg) + 'f')
            else:
                for _ in range(int(sp - od - dg - 2)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'f')


def wr_atom_list(atomic_num_f):
    atom_name_f, atom_num_f = np.zeros(0), np.zeros(0)
    atom_name_f_i, atom_num_f_i = 0, 0
    for nif in range(len(atomic_num_f)):
        if atom_name_f_i != atomic_num_f[nif]:
            atom_name_f_i = atomic_num_f[nif]
            atom_name_f = np.concatenate((atom_name_f, np.array([atom_name_f_i])), axis=0)
            atom_num_f = np.concatenate((atom_num_f, np.array([atom_num_f_i])), axis=0)
            atom_num_f_i = 1
        else:
            atom_num_f_i += 1

    atom_num_f = np.concatenate((atom_num_f, np.array([atom_num_f_i])), axis=0)
    atom_num_f = atom_num_f[1:]

    atom_name_f_str = np.zeros(atom_name_f.shape[0], dtype=object)
    for nif in range(atom_name_f.shape[0]):
        atom_name_f_str[nif] = Element.from_Z(int(atom_name_f[nif])).symbol

    return atom_name_f_str, atom_num_f


def wr_contcar(path_job, filename, title, data_lc, data_coord, atom_type, atom_num, coord_relax, velocity_f):
    atom_unique_f = np.unique(atom_type)
    atom_num_sort = np.zeros(atom_unique_f.shape[0])
    data_coord_sort = np.zeros(data_coord.shape)
    coord_relax_sort = np.zeros(coord_relax.shape)
    velocity_f_sort = np.zeros(velocity_f.shape)
    num_record_f = 0
    atom_num_expand = np.concatenate((atom_num, np.array([0])))
    for nif, nif_atom in enumerate(atom_unique_f):
        atom_match_f = atom_type == nif_atom
        atom_num_sort[nif] = np.sum(atom_num[atom_match_f])
        for nif2 in np.where(atom_match_f)[0]:
            data_coord_sort[int(num_record_f): int(num_record_f + np.sum(atom_num[nif2])), :] \
                = data_coord[int(np.sum(atom_num_expand[:nif2])): int(np.sum(atom_num_expand[:nif2 + 1])), :]
            coord_relax_sort[int(num_record_f): int(num_record_f + np.sum(atom_num[nif2])), :] \
                = coord_relax[int(np.sum(atom_num_expand[:nif2])): int(np.sum(atom_num_expand[:nif2 + 1])), :]
            velocity_f_sort[int(num_record_f): int(num_record_f + np.sum(atom_num[nif2])), :] \
                = velocity_f[int(np.sum(atom_num_expand[:nif2])): int(np.sum(atom_num_expand[:nif2 + 1])), :]
            num_record_f += np.sum(atom_num_expand[nif2])
    atom_type = atom_unique_f
    atom_num = atom_num_sort
    data_coord = data_coord_sort
    coord_relax = coord_relax_sort
    velocity_f = velocity_f_sort

    st_f = title + "\n   1.00000000000000     \n"
    for nif in range(3):
        st_f += " "
        for nif2 in range(3):
            st_f += num_printer(data_lc[nif, nif2], 22, 16, 0)
        st_f += "\n"

    for nif in range(atom_type.shape[0]):
        if len(atom_type[nif]) == 1:
            st_f += "    " + atom_type[nif]
        else:
            st_f += "   " + atom_type[nif]
    st_f += " \n"
    for nif in range(atom_num.shape[0]):
        st_f += num_printer(atom_num[nif], 7, 0, 0)

    st_f += "\nSelective dynamics\nDirect\n"
    for nif in range(data_coord.shape[0]):
        for nif2 in range(3):
            st_f += num_printer(data_coord[nif, nif2], 21, 16, 0)
        for nif2 in range(3):
            if coord_relax[nif, nif2] == 1:
                st_f += "   T"
            else:
                st_f += "   F"
        st_f += "\n"

    st_f += "\n"
    for nif in range(velocity_f.shape[0]):
        for nif2 in range(3):
            st_f += num_printer(velocity_f[nif, nif2], 16, 8, 1)
        st_f += "\n"

    with open(os.path.join(path_job, filename), 'w') as nf_rf:
        nf_rf.write(st_f)


def wr_xdatcar_head(path_job, data_lc, title, atom_type, atom_num):
    st_f = title + "\n           1\n"
    for nif in range(3):
        st_f += " "
        for nif2 in range(3):
            st_f += num_printer(data_lc[nif, nif2], 12, 6, 0)
        st_f += "\n"

    for nif in range(atom_type.shape[0]):
        if len(atom_type[nif]) == 1:
            st_f += "    " + atom_type[nif]
        else:
            st_f += "   " + atom_type[nif]
    st_f += " \n"
    for nif in range(atom_num.shape[0]):
        st_f += num_printer(atom_num[nif], 7, 0, 0)

    with open(os.path.join(path_job, "Trajectory_VASP"), 'a') as nf_rf:
        nf_rf.write(st_f + "\n")


def wr_xdatcar_coord(path_job, step_f, data_coord):
    st_f = "Direct configuration=" + num_printer(step_f, 8, 0, 0) + "\n"
    for nif in range(data_coord.shape[0]):
        st_f += " "
        for nif2 in range(3):
            st_f += num_printer(data_coord[nif, nif2], 12, 8, 0)
        st_f += "\n"

    with open(os.path.join(path_job, "Trajectory_VASP"), 'a') as nf_rf:
        nf_rf.write(st_f)


def ionic_step_printer(energy_f, d_energy, atom_coord_f, force_f, stress_f, mag_f, temp_f, volume_f, lc_mat_f):
    if np.isscalar(force_f):
        force_f_max_sca = force_f
    else:
        force_f_max_sca = np.max(np.abs(np.linalg.norm(force_f, axis=1)))

    if isinstance(energy_f, np.ndarray):  # MD
        if energy_f.size == 3:
            logfile_str_f = (num_printer(temp_f, 12, 2, 0) +
                             num_printer(volume_f, 15, 3, 0) +
                             num_printer(energy_f[0], 16, 6, 1) +
                             num_printer(d_energy[0], 10, 4, 0) +
                             num_printer(energy_f[1], 17, 6, 1) +
                             num_printer(d_energy[1], 10, 4, 0) +
                             num_printer(energy_f[2], 17, 6, 1) +
                             num_printer(d_energy[2], 10, 4, 0) +
                             num_printer(force_f_max_sca, 13, 3, 0) +
                             num_printer(np.sum(mag_f), 17, 2, 0))
            energy_f = energy_f[0]
        else:
            logfile_str_f = (num_printer(volume_f, 15, 3, 0) +
                             num_printer(energy_f, 20, 8, 1) +
                             num_printer(d_energy, 14, 6, 0) +
                             num_printer(force_f_max_sca, 18, 6, 0))
            if not np.isnan(mag_f).all():
                logfile_str_f += num_printer(np.sum(mag_f), 20, 4, 0)
    else:
        logfile_str_f = (num_printer(volume_f, 15, 3, 0) +
                         num_printer(energy_f, 20, 8, 1) +
                         num_printer(d_energy, 14, 6, 0) +
                         num_printer(force_f_max_sca, 18, 6, 0))
        if not np.isnan(mag_f).all():
            logfile_str_f += num_printer(np.sum(mag_f), 20, 4, 0)


    outcar_str_f = ("  Energy: " + num_printer(energy_f, 22, 8, 1) +
                    " eV/atoms    Volume: " + num_printer(volume_f, 16, 6, 0) +
                    " Angstrom^3\n       Lattice matrix (Angstrom)                               Stress tensor (GPa)\n")
    for nif in range(3):
        outcar_str_f += "    "
        for nif2 in range(3):
            outcar_str_f += num_printer(lc_mat_f[nif, nif2], 16, 6, 1)
        outcar_str_f += "        "
        for nif2 in range(3):
            outcar_str_f += num_printer(stress_f[nif, nif2], 16, 6, 1)
        outcar_str_f += "\n"

    if not np.isscalar(force_f):
        outcar_str_f += ("                Fractional coordinate                                "
                         "Force (eV/Angstrom)                 Magnetic moment (mu_B)\n")
        for nif in range(mag_f.shape[0]):
            for nif2 in range(3):
                outcar_str_f += num_printer(atom_coord_f[nif, nif2], 16, 6, 1)
            outcar_str_f += "    "
            for nif2 in range(3):
                outcar_str_f += num_printer(force_f[nif, nif2], 16, 6, 1)
            outcar_str_f += num_printer(mag_f[nif], 20, 6, 0) + "\n"

    return logfile_str_f, outcar_str_f


def plot_f(data_xf, data_yf, label_f, color_f, x_label_f, y_label_f, title_f, path_f, tp_f):
    if tp_f == 1:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(data_xf, data_yf, 'o-', label=label_f, color=color_f)
        ax.set_xlabel(x_label_f, fontsize=20, color="k", fontweight="bold")
        ax.set_ylabel(y_label_f, fontsize=20, color="k", fontweight="bold")
        plt.legend(fontsize=16)
        plt.title(title_f, fontsize=24, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=16)
        plt.savefig(path_f, dpi=200)
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(data_xf, data_yf, label=label_f, color=color_f)
        ax.set_xlabel(x_label_f, fontsize=20, color="k", fontweight="bold")
        ax.set_ylabel(y_label_f, fontsize=20, color="k", fontweight="bold")
        plt.legend(fontsize=16)
        plt.title(title_f, fontsize=24, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=16)
        plt.savefig(path_f, dpi=200)
        plt.close()


def cal_lc_mat(length_f, degree_f):
    degree_f *= np.pi / 180
    lc_mat = np.zeros((3, 3))
    lc_mat[0, 0] = length_f[0]
    lc_mat[1, :2] = length_f[1] * np.array([np.cos(degree_f[-1]), np.sin(degree_f[-1])])
    lc_mat[2, :2] = np.array([np.cos(degree_f[1])], (np.cos(degree_f[0]) -
                                                     np.cos(degree_f[1]) * np.cos(degree_f[-1])) / np.sin(degree_f[-1]))
    lc_mat[2, 2] = np.sqrt(1 - lc_mat[2, 0] ** 2 - lc_mat[2, 1] ** 2)
    lc_mat[2, :] *= length_f[2]

    return lc_mat


def cal_sp(model_chgnet, step_f, d_energy):
    prediction = model_chgnet.predict_structure(STRUCture)
    atom_coord_f = np.zeros((len(STRUCture), 3))
    for nif in range(atom_coord_f.shape[0]):
        atom_coord_f[nif, :] = STRUCture[nif].frac_coords
    logfile_str, outcar_str = ionic_step_printer(prediction["e"], d_energy, atom_coord_f,
                                                 prediction["f"], prediction["s"], prediction["m"],
                                                 np.nan, STRUCture[nif].lattice.volume, STRUCture[nif].lattice.matrix)

    gen(Path_log, num_printer(0, 6, 0, 0) + logfile_str, True, True, True)

    with open(Path_out, 'a') as nf_rf:
        nf_rf.write("Step " + num_printer(step_f, 8, 0, 0) + "        " + outcar_str)


def cal_md_sub(model_chgnet, path_job, nsw_md, structure, ensb, themt, temp_b, temp_e, potim, ofi_md, tau_t, tau_p, pres, bulk_m):
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import MolecularDynamics
    from pymatgen.core import Structure

    if os.path.exists(os.path.join(path_job, "CHGNet_md.log")):
        with open(os.path.join(path_job, "CHGNet_md.log"), 'w') as nf_rf:
            nf_rf.write("")

    if ensb == "npt":
        md = MolecularDynamics(atoms=structure, model=model_chgnet, ensemble=ensb, thermostat=themt,
                               starting_temperature=temp_b, temperature=temp_e, timestep=potim,
                               trajectory=os.path.join(path_job, "CHGNet_md.traj"),
                               logfile=os.path.join(path_job, "CHGNet_md.log"), loginterval=ofi_md,
                               taut=tau_t, taup=tau_p, pressure=pres, bulk_modulus=bulk_m)
    else:
        md = MolecularDynamics(atoms=structure, model=model_chgnet, ensemble=ensb, thermostat=themt,
                               starting_temperature=temp_b, temperature=temp_e, timestep=potim,
                               trajectory=os.path.join(path_job, "CHGNet_md.traj"),
                               logfile=os.path.join(path_job, "CHGNet_md.log"), loginterval=ofi_md,
                               taut=tau_t)
    md.run(nsw_md)


def cal_md_record_head(path_job, ensb, title, atom_type, atom_num):
    with open(os.path.join(path_job, "Trajectory_VASP"), 'w') as nf_rf:
        nf_rf.write("")
    if ensb != "npt":  # fix cell
        from ase.io import read
        traj = read(os.path.join(path_job, "CHGNet_md.traj"), index=0)
        lc_mat_f = cal_lc_mat(traj.cell.cellpar()[0: 3], traj.cell.cellpar()[3: 6])
        wr_xdatcar_head(path_job, lc_mat_f, title, atom_type, atom_num)


def cal_md_record(path_job, path_log, path_out, energy_hold, st_frame, ed_frame, title, atom_type, atom_num,
                  ensb, ofi_md, potim, coord_relax, plt_data_f, plt_boolean):
    for nif in range(st_frame, ed_frame):
        from ase.io import read
        nif_data = read(os.path.join(path_job, "CHGNet_md.traj"), index=nif)
        lc_mat_f = cal_lc_mat(nif_data.cell.cellpar()[0: 3], nif_data.cell.cellpar()[3: 6])
        coord_fraction_f = nif_data.get_scaled_positions()
        wr_contcar(path_job, "CONTCAR", title, lc_mat_f, coord_fraction_f,
                   atom_type, atom_num, coord_relax, nif_data.get_velocities())

        stress_mat_f = np.zeros((3, 3))
        stress_mat_f[0, 0], stress_mat_f[1, 1], stress_mat_f[2, 2] = nif_data.get_stress()[0: 3]
        stress_mat_f[1, 2], stress_mat_f[2, 1] = nif_data.get_stress()[-3], nif_data.get_stress()[-3]
        stress_mat_f[0, 2], stress_mat_f[2, 0] = nif_data.get_stress()[-2], nif_data.get_stress()[-2]
        stress_mat_f[0, 1], stress_mat_f[1, 0] = nif_data.get_stress()[-1], nif_data.get_stress()[-1]

        energy_md = np.array([nif_data.get_total_energy(), nif_data.get_kinetic_energy(),
                              nif_data.get_potential_energy()])
        energy_md /= np.sum(atom_num)
        logfile_str, outcar_str = ionic_step_printer(energy_md, energy_md - energy_hold,
                                                     coord_fraction_f, nif_data.get_forces(), stress_mat_f,
                                                     nif_data.get_magnetic_moments(), nif_data.get_temperature(),
                                                     nif_data.get_volume(), lc_mat_f)
        energy_hold = energy_md

        gen(path_log, num_printer(nif * ofi_md * potim / 1000, 10, 3, 0) +
            logfile_str, True, True, True)

        with open(path_out, 'a') as nf_rf:
            nf_rf.write("\nTime " + num_printer(nif * ofi_md * potim / 1000, 8, 3, 0) +
                        " ps      " + outcar_str)

        if ensb == "npt":  # relax cell
            wr_xdatcar_head(path_job, lc_mat_f, title, atom_type, atom_num)
        wr_xdatcar_coord(path_job, nif * ofi_md, coord_fraction_f)

        plt_data_f[nif, :] = np.array([nif * ofi_md * potim / 1000, nif_data.get_temperature(), nif_data.get_volume(),
                                       energy_md[0], energy_md[-1], -np.average(np.diag(stress_mat_f)),
                                       np.sum(nif_data.get_magnetic_moments())])

    if plt_boolean:
        # plot
        plot_f(plt_data_f[:ed_frame, 0], plt_data_f[:ed_frame, 1], "Temperature (K)", "r",
               "Simulation time (ps)", "Temperature (K)", "Temperature vs. Time",
               os.path.join(path_job, "_CHGNet_Temperature.png"), 0)
        plot_f(plt_data_f[:ed_frame, 0], plt_data_f[:ed_frame, 2], "Volume (Angstrom^3)", "k",
               "Simulation time (ps)", "Volume (Angstrom^3)", "Volume vs. Time",
               os.path.join(path_job, "_CHGNet_Volume.png"), 0)
        plot_f(plt_data_f[:ed_frame, 0], plt_data_f[:ed_frame, 3], "Total energy (eV/atoms)", "k",
               "Simulation time (ps)", "Total energy (eV/atoms)", "Total energy vs. Time",
               os.path.join(path_job, "_CHGNet_Total energy.png"), 0)
        plot_f(plt_data_f[:ed_frame, 0], plt_data_f[:ed_frame, 4],
               "Potential energy (eV/atoms)", "b",
               "Simulation time (ps)", "Potential energy (eV/atoms)", "Potential energy vs. Time",
               os.path.join(path_job, "_CHGNet_Potential energy.png"), 0)
        plot_f(plt_data_f[:ed_frame, 0], plt_data_f[:ed_frame, 3] - plt_data_f[:ed_frame, 4],
               "Kinetic energy (eV/atoms)", "r",
               "Simulation time (ps)", "Kinetic energy (eV/atoms)", "Kinetic energy vs. Time",
               os.path.join(path_job, "_CHGNet_Kinetic energy.png"), 0)
        plot_f(plt_data_f[:ed_frame, 0], plt_data_f[:ed_frame, 5] * 1E4, "Hydrostatic stress (bar)", "k",
               "Simulation time (ps)", "Hydrostatic stress (bar)", "Stress vs. Time",
               os.path.join(path_job, "_CHGNet_Hydrostatic stress.png"), 0)
        plot_f(plt_data_f[:ed_frame, 0], plt_data_f[:ed_frame, 6], "Magnetic moment (mu_B)", "k",
               "Simulation time (ps)", "Magnetic moment (mu_B)", "Magnetic moment vs. Time",
               os.path.join(path_job, "_CHGNet_Magnetic moment.png"), 0)

    return energy_hold, plt_data_f


def cal_md_record_hold(path_job, path_log, path_out, nsw_md, ensb, title,
                       atom_type, atom_num, ofi_md, potim, coord_relax, interval=10, interval_plt=3):
    time.sleep(10)
    cal_md_record_head(path_job, ensb, title, atom_type, atom_num)
    energy_hold = np.zeros(3)
    path_f = os.path.join(path_job, "CHGNet_md.log")

    last_mtime = None
    plt_pass, plt_boolean = 0, False
    current_step, new_step = 0, 0
    plt_data_f = np.zeros((int(nsw_md // ofi_md) + 1, 7))
    while True:
        if os.path.exists(path_f):
            mtime = os.path.getmtime(path_f)
            with open(path_f, 'r') as nf_rf:
                df = nf_rf.read()
                df = df.split("\n")
            new_step = len(df) - 2
            if mtime != last_mtime:
                if plt_pass > interval_plt:
                    plt_pass = 0
                    plt_boolean = True
                else:
                    plt_pass += 1
                    plt_boolean = False

                energy_hold, plt_data_f = cal_md_record(path_job, path_log, path_out, energy_hold,
                                                        current_step, new_step, title, atom_type, atom_num,
                                                        ensb, ofi_md, potim, coord_relax, plt_data_f, plt_boolean)

            last_mtime = mtime
            current_step = new_step

            if new_step * ofi_md > nsw_md:
                break
        time.sleep(interval)


def cal_opt(model_chgnet, structure_f, path_job, ibrion, ediffg, nsw, isif, ofi,
            output_contcar, output_log, output_outcar, output_xdatcar, output_diagram):
    if output_log:
        path_log = os.path.join(path_job, "CHGNet_results.log")
        with open(path_log, 'a') as nf_rf:
            nf_rf.write("# " + st_str + "\n\n")
        gen(path_log, 'Reading completed, calculation method is "Geometry Optimization".\n'
            '                           '
            'Steps      Volume             E0               dE             '
            'Force_max         Magnetic moment\n                                   '
            '(Angstrom^3)      (eV/atoms)       (eV/atoms)       (eV/Angstrom)          (mu_B)\n', False, True, True)

    original_stdout = sys.stdout
    buffer = io.StringIO()
    sys.stdout = buffer

    from chgnet.model import StructOptimizer
    relaxer = StructOptimizer(model=model_chgnet, optimizer_class=ibrion)
    result = relaxer.relax(structure_f, fmax=ediffg, steps=nsw, relax_cell=isif, loginterval=ofi)
    result_coord = result["final_structure"].frac_coords
    if output_contcar:
        atom_name_f, atom_num_f = wr_atom_list(result["final_structure"].atomic_numbers)
        if "selective_dynamics" in structure_f.site_properties:
            coord_relax_f = np.array(structure_f.site_properties["selective_dynamics"])
        else:
            coord_relax_f = np.zeros(result_coord.shape) + 1

        wr_contcar(path_job, "CONTCAR", Title, result["final_structure"].lattice.matrix,
                   result_coord, atom_name_f, atom_num_f, coord_relax_f, np.zeros(result_coord.shape))
    traj_f = result["trajectory"]
    sys.stdout = original_stdout

    with open(os.path.join(path_job, "Trajectory_VASP"), 'w') as nf_rf:
        nf_rf.write("")
    num_steps_f, num_atom_f = len(traj_f.cells), traj_f.atom_positions[0].shape[0]
    energy_f, volume_f, force_f, stress_v_f = (
        np.zeros(num_steps_f), np.zeros(num_steps_f), np.zeros(num_steps_f), np.zeros(num_steps_f))
    for nif in range(num_steps_f):
        energy_f[nif] = traj_f.energies[nif] / num_atom_f
        lc_mat_f = traj_f.cells[nif]
        volume_f[nif] = np.linalg.det(lc_mat_f)
        coord_fraction_f = np.dot(traj_f.atom_positions[nif], np.linalg.inv(lc_mat_f))
        force_f[nif] = np.max(np.linalg.norm(traj_f.forces[nif], axis=1))
        stress_f = traj_f.stresses[nif]
        stress_mat_f = np.zeros((3, 3))
        stress_mat_f[0, 0], stress_mat_f[1, 1], stress_mat_f[2, 2] = stress_f[0: 3]
        stress_mat_f[1, 2], stress_mat_f[2, 1] = stress_f[-3], stress_f[-3]
        stress_mat_f[0, 2], stress_mat_f[2, 0] = stress_f[-2], stress_f[-2]
        stress_mat_f[0, 1], stress_mat_f[1, 0] = stress_f[-1], stress_f[-1]
        stress_v_f[nif] = -np.average(stress_f[0: 3])

        logfile_str, outcar_str = ionic_step_printer(energy_f[nif], energy_f[nif] - energy_f[0],
                                                     coord_fraction_f, traj_f.forces[nif], stress_mat_f,
                                                     traj_f.magmoms[nif], np.nan, volume_f[nif], lc_mat_f)

        if output_log:
            gen(os.path.join(path_job, "CHGNet_results.log"),
                num_printer(nif * OFI_OPT, 6, 0, 0) + logfile_str, False, True, True)

        if output_outcar:
            with open(os.path.join(path_job, "Output_details"), 'a') as nf_rf:
                nf_rf.write("\nSteps " + num_printer(nif * OFI_OPT, 6, 0, 0) +
                            "      " + outcar_str)
        if output_xdatcar:
            if ISIF:  # relax cell
                wr_xdatcar_head(path_job, lc_mat_f, Title, Atom_type, Atom_num)
            wr_xdatcar_coord(path_job, nif * OFI_OPT, coord_fraction_f)

    if output_diagram:
        # plot
        step_f = np.arange(0, num_steps_f, 1)
        plot_f(step_f, volume_f, "Volume (Angstrom^3)", "k",
               "Steps", "Volume (Angstrom^3)", "Volume vs. Step",
               os.path.join(path_job, "_CHGNet_Optimization_Volume.png"), 0)
        plot_f(step_f, stress_v_f * 1E4, "Hydrostatic stress (Bar)", "k",
               "Steps", "Hydrostatic stress (Bar)", "Hydrostatic stress vs. Step",
               os.path.join(path_job, "_CHGNet_Optimization_Hydrostatic stress.png"), 0)

        fig, ax1 = plt.subplots(figsize=(16, 9))
        ax1.plot(step_f, energy_f - energy_f[0], label="Total energy difference (eV/atoms)", color="r")
        ax1.set_xlabel("Steps", fontsize=20, color="k", fontweight="bold")
        ax1.set_ylabel("Total energy difference (eV/atoms)", fontsize=20, color="r", fontweight="bold")
        plt.legend(fontsize=16)
        ax2 = ax1.twinx()
        ax2.plot(step_f, force_f, label="Force_max (eV/Angstrom)", color="b")
        ax2.set_ylabel("Force_max (eV/Angstrom)", fontsize=20, color="b", fontweight="bold")
        ax2.set_yscale("log")
        plt.title("Energy & Force vs. Steps", fontsize=24, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=16)
        plt.savefig(os.path.join(path_job, "_CHGNet_Optimization_Energy_Force.png"), dpi=200)
        plt.close()

    result_structure_f = result["final_structure"]
    if "selective_dynamics" in result["final_structure"].site_properties:
        result_structure_f.site_properties["selective_dynamics"] = coord_relax_f
    else:
        result_structure_f.add_site_property("selective_dynamics", coord_relax_f)

    return result["trajectory"].energies[-1], result_structure_f


def wr_freq(path_gaussian, structure_origin_f, coord_cartesian, relax_atom_f, data_force, data_num, data_vec):
    atom_type = structure_origin_f.atomic_numbers
    atom_num_sum = coord_cartesian.shape[0]
    data_freq = data_num / c_cons / 100
    data_force *= 0.529177249 / 27.21138602
    with open(path_gaussian, 'w') as nf:
        nf.write(" GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n" +
                 " Number of steps in this run=   2 maximum allowed number of steps=   2.\n" +
                 " GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n" +
                 "                         Standard orientation:                         \n" +
                 " ---------------------------------------------------------------------\n" +
                 " Center     Atomic      Atomic             Coordinates (Angstroms)\n" +
                 " Number     Number       Type             X           Y           Z\n" +
                 " ---------------------------------------------------------------------\n")

    nif_str = ""
    for nif in range(atom_num_sum):
        nif_str += (num_printer(nif + 1, 8, 0, 0) +
                    num_printer(atom_type[nif], 12, 0, 0) + "           0    ")

        for nif2 in range(3):
            nif_str += num_printer(coord_cartesian[nif, nif2], 12, 6, 0)
        nif_str += "\n"
    for nif in range(3):
        nif_str += num_printer(atom_num_sum + nif + 1, 8, 0, 0) + "         -2           0    "
        for nif2 in range(3):
            nif_str += num_printer(LC[nif, nif2], 12, 6, 0)
        nif_str += "\n"

    with open(path_gaussian, 'a') as nf:
        nf.write(nif_str)
        nf.write(" ---------------------------------------------------------------------\n" +
                 " Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering\n" +
                 " activities (A**4/AMU), depolarization ratios for plane and unpolarized\n" +
                 " incident light, reduced masses (AMU), force constants (mDyne/A),\n" + " and normal coordinates:\n")

    dof_seq = np.linspace(np.sum(relax_atom_f) - 1, 0, int(np.sum(relax_atom_f)))
    nif_str = ""
    for nif in range(int(np.ceil(np.sum(relax_atom_f) / 3))):
        dof_seq_sel = dof_seq[: min(3, np.shape(dof_seq)[0])]
        dof_seq = dof_seq[min(3, np.shape(dof_seq)[0]):]
        for nif2 in range(np.shape(dof_seq_sel)[0]):
            if nif2 == 0:
                nif_str += num_printer(3 * nif + 1 + nif2, 23, 0, 0)
            else:
                nif_str += num_printer(3 * nif + 1 + nif2, 24, 0, 0)
        nif_str += "\n Frequencies --"

        for nif2 in range(np.shape(dof_seq_sel)[0]):
            if nif2 == 0:
                nif_str += num_printer(data_freq[-3 * nif - nif2 - 1], 11, 4, 0)
            else:
                nif_str += num_printer(data_freq[-3 * nif - nif2 - 1], 23, 4, 0)
        nif_str += "\n IR Inten    --     0.0000                 0.0000                 0.0000\n  Atom  AN"
        for nif2 in range(np.shape(dof_seq_sel)[0]):
            nif_str += "      X      Y      Z  "
        nif_str += "\n"

        for nif2 in range(atom_num_sum):
            nif_str += num_printer(nif2 + 1, 7, 0, 0) + num_printer(atom_type[nif2], 5, 0, 0)
            for nif3 in range(np.shape(dof_seq_sel)[0]):
                nif_str += "  "
                for nif4 in range(3):
                    nif_str += num_printer(data_vec[-3 * nif - nif3 - 1, nif2, nif4], 7, 2, 0)
            nif_str += "\n"
    with open(path_gaussian, 'a') as nf:
        nf.write(nif_str)

    nif_str = ("\n  ***** Axes restored to original set *****\n" +
               "  -------------------------------------------------------------------\n" +
               "  Center     Atomic                   Forces (Hartrees/Bohr)\n" +
               "  Number     Number              X              Y              Z\n" +
               "  -------------------------------------------------------------------\n")
    for nif in range(atom_num_sum):
        nif_str += (num_printer(nif + 1, 8, 0, 0) +
                    num_printer(atom_type[nif], 12, 0, 0) + "      ")

        for nif2 in range(3):
            nif_str += num_printer(data_force[nif, nif2], 15, 9, 0)
        nif_str += "\n"
    nif_str += ("\n  -------------------------------------------------------------------\n\n" +
                " GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n " +
                " Step number   1 out of a maximum of   2\n" +
                " GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n")
    with open(path_gaussian, 'a') as nf:
        nf.write(nif_str)


def cal_phonon(path_job, model_chgnet, structure_origin_f, temp_sta,
               output_progress, output_progress_detail, output_thermal_results, output_gaussview, output_spectra):
    if "selective_dynamics" in structure_origin_f.site_properties:
        relax_atom_mat = np.array(structure_origin_f.site_properties["selective_dynamics"]) == 1
    else:
        relax_atom_mat = np.zeros(structure_origin_f.frac_coords.shape)
        relax_atom_mat = relax_atom_mat == 0

    prediction_origin_f = model_chgnet.predict_structure(structure_origin_f, task="ef")
    num_atom = relax_atom_mat.shape[0]
    energy_corr = prediction_origin_f["e"] * num_atom

    modify_mat = np.eye(3) * POTIM_PH
    lc_inv_f = np.linalg.inv(structure_origin_f.lattice.matrix)
    modify_position_frac = np.dot(modify_mat, lc_inv_f)

    dof_f = int(np.sum(relax_atom_mat))
    mass_array_f = np.zeros(dof_f)
    hessian_mat_f = np.zeros((mass_array_f.shape[0], mass_array_f.shape[0]))
    relax_seq = 0
    cp_f = 0
    if output_progress & output_progress_detail:
        gen(Path_log, "Current progress for phonon calculation", True, True, True)
        gen(Path_log, "0   10   20   30   40   50   60   70   80   90   100\n"
                      "                         "
                      "|----|----|----|----|----|----|----|----|----|----|\n"
                      "                         ", True, True, False)
    for nif_atom in range(relax_atom_mat.shape[0]):
        for nif_xyz in range(3):
            if relax_atom_mat[nif_atom, nif_xyz]:
                mass_array_f[relax_seq] = Element.from_Z(structure_origin_f.atomic_numbers[nif_atom]).atomic_mass

                structure_p_f = structure_origin_f.copy()
                structure_p_f.replace(nif_atom, structure_origin_f.sites[nif_atom].specie,
                                      structure_origin_f.frac_coords[nif_atom, :] + modify_position_frac[nif_xyz, :])

                structure_n_f = structure_origin_f.copy()
                structure_n_f.replace(nif_atom, structure_origin_f.sites[nif_atom].specie,
                                      structure_origin_f.frac_coords[nif_atom, :] - modify_position_frac[nif_xyz, :])

                prediction_f = model_chgnet.predict_structure([structure_p_f, structure_n_f],
                                                              task="ef", batch_size=2)
                der_force = (prediction_f[0]["f"] - prediction_f[1]["f"]) / (2 * POTIM_PH)
                hessian_mat_f[relax_seq, :] = der_force[relax_atom_mat]
                relax_seq += 1

                if cp_f < relax_seq:
                    if output_progress:
                        if output_progress_detail:
                            gen(Path_log, "*", True, False, False)
                            cp_f += dof_f / 50
                        else:
                            gen(Path_log, "*", True, False, False)
                            cp_f += dof_f / 10

    if output_progress * output_progress_detail:
        gen(Path_log, "\n", True, True, False)

    reduce_mass_mat_f = np.sqrt(mass_array_f[:, np.newaxis] * mass_array_f[np.newaxis, :])  # in amu
    hessian_mat_f /= -reduce_mass_mat_f
    eigen_val, eigen_vec = np.linalg.eig(hessian_mat_f)
    eigen_val = q_cons * 1E20 / AMU_cons * np.real(eigen_val)  # in Hz**2

    eigen_vec = np.real(eigen_vec)
    eigen_vec_mat = np.zeros((dof_f, relax_atom_mat.shape[0], relax_atom_mat.shape[1]))
    nif_record = 0
    for nif_atom in range(relax_atom_mat.shape[0]):
        for nif_xyz in range(relax_atom_mat.shape[1]):
            if relax_atom_mat[nif_atom, nif_xyz]:
                eigen_vec_mat[:, nif_atom, nif_xyz] = eigen_vec[nif_record, :]
                nif_record += 1

    eigen_vec_mat /= np.sqrt(np.max(np.linalg.norm(eigen_vec_mat, axis=-1), axis=1))[:, np.newaxis, np.newaxis]

    vib_freq_f = np.sqrt(np.abs(eigen_val)) / (2 * np.pi)
    vib_freq_f = np.where(eigen_val < 0, -vib_freq_f, vib_freq_f)  # in Hz

    eigen_vec_mat = eigen_vec_mat[np.argsort(-vib_freq_f)]
    vib_freq_f = -np.sort(-vib_freq_f)  # in Hz

    num_image_freq_f = int(np.sum(vib_freq_f < 0))
    num_real_low_freq_f = int(np.sum(vib_freq_f < np.zeros(dof_f) + CUT_WN * c_cons * 100) - num_image_freq_f)
    vib_freq_f_save = np.maximum(vib_freq_f, np.zeros(dof_f) + CUT_WN * c_cons * 100)  # in Hz
    vib_energy = h_cons * vib_freq_f_save / q_cons  # in eV
    vib_zpe_f = np.sum(vib_energy) / 2  # in eV

    vib_internal_energy_f = np.sum(vib_energy / (np.exp(vib_energy * q_cons / (kB_cons * temp_sta)) - 1))

    vib_temp_ratio_f = h_cons * vib_freq_f_save / (kB_cons * temp_sta)
    vib_entropy_f = kB_cons / q_cons * np.sum(vib_temp_ratio_f / (np.exp(vib_temp_ratio_f) - 1) -
                                              np.log(1 - np.exp(-vib_temp_ratio_f)))  # in eV / K

    if output_thermal_results:
        gen(Path_log, "", True, True, True)
        gen(Path_log, "The phonon calculation is complete. A total of " +
            format(dof_f - num_image_freq_f, '.0f') + " real frequencies and " +
            format(num_image_freq_f, '.0f') + " imaginary frequencies were found.",
            True, True, True)
        gen(Path_log, "The program corrects the " + format(num_real_low_freq_f + num_image_freq_f, '.0f') +
            " frequencies with wave numbers below " + format(CUT_WN, '.1f') +
            " cm-1 to obtain the thermodynamic quantities at " + format(temp_sta, '.1f') +
            " K:\n", True, True, True)

        energy_corr += vib_zpe_f
        gen(Path_log, "Zero-point energy ZPE     " +
            num_printer(vib_zpe_f * q_cons * NA_cons / kcal_to_J_cons, 15, 3, 0) + " kcal/mol, " +
            num_printer(vib_zpe_f, 10, 4, 0) + " eV" +
            num_printer(vib_zpe_f / num_atom, 14, 6, 0) +
            " eV/atoms\n                         Corrected energy, E" +
            num_printer(energy_corr, 43, 3, 0) + " eV" +
            num_printer(energy_corr / num_atom, 14, 6, 0) + " eV/atoms",
            True, True, True)
        energy_corr += vib_internal_energy_f
        gen(Path_log, "Thermal correction to U(T)" +
            num_printer((vib_zpe_f + vib_internal_energy_f) *
                        q_cons * NA_cons / kcal_to_J_cons, 15, 3, 0) + " kcal/mol, " +
            num_printer(vib_zpe_f + vib_internal_energy_f, 10, 4, 0) + " eV" +
            num_printer((vib_zpe_f + vib_internal_energy_f) / num_atom, 14, 6, 0) +
            " eV/atoms\n                         Corrected energy, U" +
            num_printer(energy_corr, 43, 3, 0) + " eV" +
            num_printer(energy_corr / num_atom, 14, 6, 0) + " eV/atoms",
            True, True, True)
        energy_corr -= vib_entropy_f * temp_sta
        gen(Path_log, "Thermal correction to G(T)" +
            num_printer((vib_zpe_f + vib_internal_energy_f - vib_entropy_f * temp_sta) *
                        q_cons * NA_cons / kcal_to_J_cons, 15, 3, 0) + " kcal/mol, " +
            num_printer(vib_zpe_f + vib_internal_energy_f -
                        vib_entropy_f * temp_sta, 10, 4, 0) + " eV" +
            num_printer((vib_zpe_f + vib_internal_energy_f -
                         vib_entropy_f * temp_sta) / num_atom, 14, 6, 0) +
            " eV/atoms\n                         Corrected energy, G" +
            num_printer(energy_corr, 43, 3, 0) + " eV" +
            num_printer(energy_corr / num_atom, 14, 6, 0) + " eV/atoms",
            True, True, True)
        gen(Path_log, "Entropy S                 " +
            num_printer(vib_entropy_f * q_cons * NA_cons / 1000, 15, 3, 0) + " kJ/mol  , " +
            num_printer(vib_entropy_f, 10, 4, 0) + " eV/K" +
            num_printer(vib_entropy_f / num_atom, 12, 6, 0) + " eV/atoms/K",
            True, True, True)
        gen(Path_log, "Entropy contribution T*S  " +
            num_printer(vib_entropy_f * temp_sta * q_cons * NA_cons / 1000, 15, 3, 0) + " kJ/mol  , " +
            num_printer(vib_entropy_f * temp_sta, 10, 4, 0) + " eV" +
            num_printer(vib_entropy_f * temp_sta / num_atom, 14, 6, 0) + " eV/atoms",
            True, True, True)

    if output_gaussview:
        wr_freq(os.path.join(path_job, "Phonon_GaussView.log"), structure_origin_f,
                np.dot(structure_origin_f.frac_coords, structure_origin_f.lattice.matrix), relax_atom_mat,
                prediction_origin_f["f"], vib_freq_f, eigen_vec_mat)

    if output_spectra:
        st_f = "    Frequency (THz)    Wave number (cm-1)    Energy (meV)\n"
        for nif_vib in vib_freq_f[::-1]:
            st_f += (num_printer(nif_vib / 1E12, 15, 6, 0) +
                     num_printer(nif_vib / c_cons / 100, 18, 3, 0) +
                     num_printer(nif_vib * h_cons / q_cons * 1E3, 19, 4, 0) + "\n")
        with open(os.path.join(path_job, "_CHGNet_Phonon.log"), "w") as nf_rf:
            nf_rf.write(st_f)

        gau_sd_f = (GAU_FWHM / (2 * np.sqrt(2 * np.log(2)))) * c_cons * 100
        spectra_freq_f = np.arange(vib_freq_f[-1] - 3 * gau_sd_f, vib_freq_f[0] + 4 * gau_sd_f, gau_sd_f / 10)
        spectra_dos_f = np.zeros(spectra_freq_f.shape[0])
        for nif_vib in vib_freq_f:  # in Hz
            spectra_dos_f += np.exp(-1 * (spectra_freq_f - nif_vib) ** 2 / (2 * gau_sd_f ** 2))
        spectra_dos_f /= np.sqrt(2 * np.pi) * gau_sd_f

        st_f = ("    Frequency (THz)    Density of states (THz-1)    "
                "    Wave number (cm-1)    Density of states (cm)    "
                "    Energy (meV)    Density of states (meV-1)\n")
        for nif in range(spectra_freq_f.shape[0]):
            st_f += (num_printer(spectra_freq_f[nif] / 1E12, 15, 6, 0) +
                     num_printer(spectra_dos_f[nif] * 1E12, 22, 8, 0) +
                     num_printer(spectra_freq_f[nif] / c_cons / 100, 30, 3, 0) +
                     num_printer(spectra_dos_f[nif] * c_cons * 100, 25, 8, 0) +
                     num_printer(spectra_freq_f[nif] * h_cons / q_cons * 1E3, 25, 4, 0) +
                     num_printer(spectra_dos_f[nif] / h_cons * q_cons / 1E3, 21, 8, 0) + "\n")
        with open(os.path.join(path_job, "_CHGNet_Phonon_distribution.log"), "w") as nf_rf:
            nf_rf.write(st_f)

        plot_f(spectra_freq_f / 1E12, spectra_dos_f * 1E12,
               "DOS", "b", "Frequency (THz)", "Density of states (THz-1)",
               "Phonon - Standard deviation: " + format(gau_sd_f / 1E12, '.2f') + " THz",
               os.path.join(path_job, "_CHGNet_Phonon_THz.png"), 0)
        plot_f(spectra_freq_f / c_cons / 100, spectra_dos_f * c_cons * 100,
               "DOS", "r", "Wave number (cm-1)", "Density of states (cm)",
               "Phonon - Standard deviation: " + format(gau_sd_f / c_cons / 100, '.1f') + " cm-1",
               os.path.join(path_job, "_CHGNet_Phonon_cm-1.png"), 0)
        plot_f(spectra_freq_f * h_cons / q_cons * 1E3, spectra_dos_f / h_cons * q_cons / 1E3,
               "DOS", "k", "Energy (meV)", "Density of states (meV-1)",
               "Phonon - Standard deviation: " + format(gau_sd_f * h_cons / q_cons * 1E3, '.2f') + " meV-1",
               os.path.join(path_job, "_CHGNet_Phonon_meV.png"), 0)

    return (int(dof_f - num_real_low_freq_f - num_image_freq_f), num_real_low_freq_f, num_image_freq_f,
            vib_freq_f, eigen_vec_mat, vib_zpe_f, vib_internal_energy_f, vib_entropy_f)


def select_index(atom_doped_mat_f):
    index_s1 = np.random.randint(FASEQ_array_index.shape[0])  # choose 1st index
    filtered_index_s1 = np.delete(np.arange(0, FASEQ_array_index.shape[0], 1),
                                  np.where(atom_doped_mat_f == atom_doped_mat_f[index_s1])[0])
    # filtered_index_s1: list the index without 1st kind of element
    index_s2 = np.random.choice(filtered_index_s1)
    element_s1_f = atom_doped_mat_f[index_s2]
    atom_doped_mat_f[index_s2] = atom_doped_mat_f[index_s1]
    atom_doped_mat_f[index_s1] = element_s1_f

    return atom_doped_mat_f


def replace_atom(struc_ori, atom_doped_mat, faseq_array_index):
    # atom_doped_mat: [Co, Ni, Mn, Ni, Va, Co, ...]
    # faseq_array_index: [3, 2, 4, 5, 1, 9, ...]
    struc_f = struc_ori.copy()
    vacancy_site = np.zeros(np.sum(atom_doped_mat == "Va")) - 1
    for nif, nif_index in enumerate(faseq_array_index):
        if atom_doped_mat[nif] == "Va":
            vacancy_site[np.sum(vacancy_site != -1)] = nif_index
        else:
            struc_f.replace(idx=int(nif_index), species=atom_doped_mat[nif])

    for nif_key, nif_prop in STRUCture_site_prop.items():
        struc_f.add_site_property(nif_key, nif_prop)

    struc_f.remove_sites(vacancy_site.astype(int).tolist())

    return struc_f


def replace_atom_beginning(struc_ori, faseq_array_index):
    atom_doped_label_mat = np.zeros((len(NAME_OCC), 2), dtype=object)
    for nif in range(atom_doped_label_mat.shape[0]):
        atom_doped_label_mat[nif, 0] = NAME_OCC[nif]
        atom_doped_label_mat[nif, 1] = int(NUM_OCC[nif])
    atom_doped_mat = np.zeros(int(np.sum(atom_doped_label_mat[:, 1])), dtype=object)
    for nif in range(len(NAME_OCC)):
        atom_doped_mat[np.sum(atom_doped_label_mat[: nif, 1]):
                       np.sum(atom_doped_label_mat[: nif + 1, 1])] = atom_doped_label_mat[nif, 0]

    struc_f = replace_atom(struc_ori, atom_doped_mat, faseq_array_index)

    return struc_f, atom_doped_mat


def get_structure_hash(structure_f):
    structure_sort_f = structure_f.get_sorted_structure()
    sites_f = structure_sort_f.sites
    sorted_sites_f = sorted(sites_f, key=lambda
        site: (site.specie.symbol, site.frac_coords[0], site.frac_coords[1], site.frac_coords[2]))
    new_structure_f = Structure(structure_sort_f.lattice, [site.species for site in sorted_sites_f],
                                [site.frac_coords for site in sorted_sites_f])

    return hashlib.sha256(str(new_structure_f.as_dict()).encode()).hexdigest()


def cal_mc(path_log, structure_f):
    structure_origin_f = structure_f.copy()
    structure_save_f, structure_screen_f = {}, {}
    st_time_mc = time.time()
    structure_f, atom_doped_mat_f = replace_atom_beginning(structure_origin_f, FASEQ_array_index)  # initialization
    num_occ_f = int(FASEQ_array.shape[0] - np.sum(atom_doped_mat_f == "Va"))

    path_res_sub = os.path.join(Path_tmp, "_Initial")
    if not os.path.exists(path_res_sub):
        os.mkdir(path_res_sub)
    energy_save_mat_f = np.zeros((int(NSW_MC_G), int(NSW_MC_S) + 1))
    energy_save_f = np.zeros((int(NSW_MC_G), int(NSW_MC_S) + 1))

    atom_name_mc_f, atom_num_mc_f = wr_atom_list(structure_f.atomic_numbers)
    relax_atom_f = np.zeros(structure_f.frac_coords.shape) + 1
    if "selective_dynamics" in structure_f.site_properties:
        relax_atom_f = np.array(structure_f.site_properties["selective_dynamics"])
    wr_contcar(path_res_sub, "POSCAR", Title, structure_f.lattice.matrix, structure_f.frac_coords,
               atom_name_mc_f, atom_num_mc_f, relax_atom_f, np.zeros((structure_f.frac_coords.shape[0], 3)))
    energy_initial_f, _ = (
        cal_opt(Model_CHGNET, structure_f, path_res_sub, IBRION, EDIFFG1, NSW_OPT1, ISIF1, OFI_OPT1,
                True, True, False, False, False))
    time_cost_mc = time.time() - st_time_mc  # in sec
    time_cost_total = time_cost_mc * (1 + NSW_MC_G * (NSW_MC_S + NSW_OPT2 / NSW_OPT1 * SCQU1) +
                                      SCQU2 * NSW_MCCE * (NSW_OPT3 / NSW_OPT1 + 6 * num_occ_f))  # in sec
    gen(path_log,
        "Monte Carlo Sampling: Initial configuration, it takes " +
        num_printer(time_cost_mc / 60, 4, 2, 0) + " mins\n"
                                                  "                         The program is expected to take " +
        num_printer(time_cost_total / 3600, np.ceil(np.log10(time_cost_total / 3600) + 1), 1, 0) +
        " hrs to run, ending at " +
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st_time_mc + time_cost_total)) +
        "\n\n                            Group-Iteration           Energy (eV/atoms)            dE (eV)   "
        "      Acceptance rate            State\n", True, True, True)
    energy_save_mat_f[:, 0] = energy_initial_f / num_occ_f
    energy_save_f[:, 0] = energy_initial_f / num_occ_f

    hash_mat_f = np.zeros((NSW_MC_G, NSW_MC_S + 1), dtype=object)
    hash_mat_f[:, 0] = get_structure_hash(structure_f)
    atom_doped_mat_save_f = np.zeros((int(NSW_MC_G), int(NSW_MC_S) + 1,
                                      FASEQ_array_index.shape[0]), dtype=object)
    atom_doped_mat_save_f[:, 0, :] = atom_doped_mat_f[np.newaxis, :]
    convg_num_f = int(min([max([NSW_MC_S / 4, 5]), 250]))
    # 1st screening
    for nif_main in range(int(NSW_MC_G)):  # main loop
        structure_save_f[nif_main] = {}
        path_res_main = os.path.join(Path_tmp, "tmp_" + str(nif_main + 1))
        if not os.path.exists(path_res_main):
            os.mkdir(path_res_main)

        structure_f, atom_doped_mat_f = replace_atom_beginning(structure_origin_f, FASEQ_array_index)  # initialization
        structure_save_f[nif_main][0] = structure_f

        structure_find_index_f, structure_find_num_f, structure_find_num_save_f = 0, 1, np.array([0])
        for nif_sub in range(1, int(NSW_MC_S) + 1):  # minor loop
            path_res_sub = os.path.join(path_res_main, "tmp_" + str(nif_main + 1) + "_" + str(nif_sub))
            if not os.path.exists(path_res_sub):
                os.mkdir(path_res_sub)

            atom_doped_mat_old_f = atom_doped_mat_f.copy()
            # get the "POSCAR"
            atom_doped_mat_f = select_index(atom_doped_mat_f)
            atom_doped_mat_save_f[nif_main, nif_sub] = atom_doped_mat_f
            structure_f = replace_atom(structure_origin_f, atom_doped_mat_f, FASEQ_array_index)
            structure_save_f[nif_main][nif_sub] = structure_f
            hash_v = get_structure_hash(structure_f)
            if np.sum(hash_v == hash_mat_f) != 0:  # has been calculated
                index_hash_f = np.argwhere(hash_v == hash_mat_f)[0, :]
                energy_save_mat_f[nif_main, nif_sub] = energy_save_mat_f[index_hash_f[0], index_hash_f[1]]
                pre_cal_state_f = " (Precomputation)"
            else:
                # calculate the energy
                atom_name_mc_f, atom_num_mc_f = wr_atom_list(structure_f.atomic_numbers)
                relax_atom_f = np.zeros(structure_f.frac_coords.shape) + 1
                if "selective_dynamics" in structure_f.site_properties:
                    relax_atom_f = np.array(structure_f.site_properties["selective_dynamics"])
                wr_contcar(path_res_sub, "POSCAR", Title, structure_f.lattice.matrix, structure_f.frac_coords,
                           atom_name_mc_f, atom_num_mc_f, relax_atom_f, np.zeros((structure_f.frac_coords.shape[0], 3)))
                energy_system_f, _ = (
                    cal_opt(Model_CHGNET, structure_f, path_res_sub, IBRION, EDIFFG1, NSW_OPT1, ISIF1, OFI_OPT1,
                            True, True, False, False, False))
                energy_save_mat_f[nif_main, nif_sub] = energy_system_f / num_occ_f
                pre_cal_state_f = "                 "

            hash_mat_f[nif_main, nif_sub] = hash_v

            delta_e_f = energy_save_mat_f[nif_main, nif_sub] - energy_save_mat_f[nif_main, structure_find_index_f]
            delta_e_joule = delta_e_f * num_occ_f * q_cons
            acceptance_rate_f = min([1, np.exp(-delta_e_joule / kB_cons / TEMP_MC)])
            if acceptance_rate_f >= np.random.rand():
                state_mc_f = "Acceptance"
                structure_find_index_f = nif_sub
                energy_save_f[nif_main, structure_find_num_f] = energy_save_mat_f[nif_main, nif_sub]
                structure_find_num_f += 1
                structure_find_num_save_f = (
                    np.concatenate((structure_find_num_save_f, np.array([nif_sub])), axis=0))

                plot_f(np.arange(0, structure_find_num_f, 1),
                       (energy_save_f[nif_main, :structure_find_num_f] - energy_save_f[nif_main, 0]) * num_occ_f,
                       "Accepted energy", "r",
                       "Accepted steps", "Energy difference (eV)", "",
                       os.path.join(Path_tmp, "Energy_" + format(nif_main + 1, '.0f') + "_accepted.png"), 1)
                plot_f(structure_find_num_save_f,
                       (energy_save_f[nif_main, :structure_find_num_f] - energy_save_f[nif_main, 0]) * num_occ_f,
                       "Accepted energy", "b",
                       "Steps", "Energy difference (eV)", "",
                       os.path.join(Path_tmp, "Energy_" + format(nif_main + 1, '.0f') + ".png"), 1)
            else:
                state_mc_f = "*Rejection"
                atom_doped_mat_f = atom_doped_mat_old_f

            gen(path_log,
                num_printer(nif_main + 1, 8, 0, 0) + " - " +
                num_printer(nif_sub, 8, 0, 0) +
                num_printer(energy_save_mat_f[nif_main, nif_sub], 24, 6, 0) +
                num_printer(delta_e_f * num_occ_f, 23, 3, 0) +
                num_printer(acceptance_rate_f * 100, 20, 2, 0) + "%              " +
                state_mc_f + pre_cal_state_f, True, True, True)

            if nif_sub > convg_num_f:
                if structure_find_num_f > 5:
                    std_energy_f = np.std(energy_save_f[-5:])
                    if std_energy_f <= EDIFFG0:
                        gen(path_log, "Reaching convergence conditions.", True, True, True)
                        break

        path_plt_f = os.path.join(Path_tmp, "Energy_" + format(nif_main + 1, '.0f') + "_accepted.png")
        if os.path.exists(path_plt_f):
            os.remove(path_plt_f)
        plot_f(np.arange(0, structure_find_num_f, 1),
               (energy_save_f[nif_main, :structure_find_num_f] - energy_save_f[nif_main, 0]) * num_occ_f,
               "Accepted energy", "r",
               "Accepted steps", "Energy difference (eV)", "",
               os.path.join(Path_tmp, "Energy_" + format(nif_main + 1, '.0f') + "__N_" +
                            format(structure_find_num_f, '.0f') + "_accepted.png"), 1)

        path_plt_f = os.path.join(Path_tmp, "Energy_" + format(nif_main + 1, '.0f') + ".png")
        if os.path.exists(path_plt_f):
            os.remove(path_plt_f)
        plot_f(structure_find_num_save_f,
               (energy_save_f[nif_main, :structure_find_num_f] - energy_save_f[nif_main, 0]) * num_occ_f,
               "Accepted energy", "b",
               "Steps", "Energy difference (eV)", "",
               os.path.join(Path_tmp, "Energy_" + format(nif_main + 1, '.0f') + "__N_" +
                            format(structure_find_num_f, '.0f') + ".png"), 1)

        # choose the configurations with low energies
        energy_save_array_f = energy_save_mat_f[nif_main, :].copy()
        structure_screen_f[nif_main] = {}
        for nif_scr in range(int(SCQU1)):
            index_min_f = int(np.argmin(energy_save_array_f))
            structure_screen_f[nif_main][nif_scr] = structure_save_f[nif_main][index_min_f]
            energy_save_array_f[index_min_f] = np.inf

        gen(path_log,
            "Monte Carlo sampling of group " + num_printer(nif_main + 1, NSW_MC_G_dight, 0, 0) +
            " ends with a minimum energy of " + num_printer(np.min(energy_save_mat_f[nif_main, :]), 8, 4, 0) +
            " eV/atoms in structures " + num_printer(nif_main + 1, NSW_MC_G_dight, 0, 0) + "-" +
            num_printer(np.where(energy_save_mat_f[nif_main, :] - energy_save_mat_f[nif_main, :].min() < 1E-6)[0][-1],
                        NSW_MC_S_dight, 0, 0) + "\n", True, True, True)

        shutil.make_archive(path_res_main, 'gztar', path_res_main)
        if os.path.exists(path_res_main):
            shutil.rmtree(path_res_main)

    return num_occ_f, energy_save_mat_f, structure_screen_f, atom_doped_mat_save_f


def cal_mc_opt(path_log, path_tmp):
    gen(path_log, "", True, True, True)
    gen(path_log, "Start secondary screening\n"
                  "                            Group/Iteration"
                  "           Energy (eV/atoms)           Structure\n", True, True, True)

    energy_screen_mat_f = np.zeros((int(NSW_MC_G), int(SCQU1)))
    index_screen_mat_f = np.zeros((int(NSW_MC_G), int(SCQU1)))
    num_occ_f = int(FASEQ_array.shape[0] - np.sum(Atom_doped_mat[0, 0, :] == "Va"))
    struc_screen_f = {}
    for nif_main in range(energy_screen_mat_f.shape[0]):
        path_res_main = os.path.join(path_tmp, "tmp_screening")
        if not os.path.exists(path_res_main):
            os.mkdir(path_res_main)

        struc_screen_f[nif_main] = {}
        for nif_scr in range(energy_screen_mat_f.shape[1]):
            nif_sub = np.argsort(Energy_save_mat[nif_main, :])[nif_scr]
            path_res_sub = os.path.join(path_res_main, "tmp_" + str(nif_main + 1) + "_" + str(nif_sub))
            if not os.path.exists(path_res_sub):
                os.mkdir(path_res_sub)
            index_screen_mat_f[nif_main, nif_scr] = nif_sub

            struc_f = STRUCture_screen[nif_main][nif_scr]
            atom_name_mc_f, atom_num_mc_f = wr_atom_list(struc_f.atomic_numbers)
            relax_atom_f = np.zeros(struc_f.frac_coords.shape) + 1
            if "selective_dynamics" in struc_f.site_properties:
                relax_atom_f = np.array(struc_f.site_properties["selective_dynamics"])
            wr_contcar(path_res_sub, "POSCAR", Title, struc_f.lattice.matrix, struc_f.frac_coords,
                       atom_name_mc_f, atom_num_mc_f, relax_atom_f, np.zeros((struc_f.frac_coords.shape[0], 3)))
            energy_system_f, struc_screen_i_f = (
                cal_opt(Model_CHGNET, struc_f, path_res_sub, IBRION, EDIFFG2, NSW_OPT2, ISIF2, OFI_OPT2,
                        True, True, True, True, True))
            energy_screen_mat_f[nif_main, nif_scr] = energy_system_f / Num_occ
            struc_screen_f[nif_main][nif_scr] = struc_screen_i_f

            gen(path_log,
                num_printer(nif_main + 1, 8, 0, 0) + " / " +
                num_printer(nif_scr + 1, 8, 0, 0) +
                num_printer(energy_screen_mat_f[nif_main, nif_scr], 24, 6, 0) +
                num_printer(nif_main + 1, 19 + NSW_MC_G_dight, 0, 0) + "-" +
                num_printer(nif_sub, NSW_MC_S_dight, 0, 0), True, True, True)

    gen(path_log, "", True, True, True)
    num_screened_f = min([int(SCQU2), int(NSW_MC_G * SCQU1)])
    energy_screened_f = np.zeros(num_screened_f)
    atom_doped_mat_screened_f = np.zeros((num_screened_f, FASEQ_array_index.shape[0]), dtype=object)
    energy_screen_mat_min_f = np.min(energy_screen_mat_f)
    for nif_scr in range(energy_screened_f.shape[0]):
        energy_screened_f[nif_scr] = np.min(energy_screen_mat_f)

        index_min_f = np.argmin(energy_screen_mat_f)
        index_min_f = np.unravel_index(index_min_f, energy_screen_mat_f.shape)
        index_min_2nd_f = int(index_screen_mat_f[index_min_f[0], index_min_f[1]])
        atom_doped_mat_screened_f[nif_scr, :] = Atom_doped_mat[int(index_min_f[0]), index_min_2nd_f, :]

        relative_energy_i_f = (energy_screen_mat_f[index_min_f] - energy_screen_mat_min_f) * num_occ_f
        path_index_f = os.path.join(path_tmp, "tmp_screening", "tmp_" +
                                    str(index_min_f[0] + 1) + "_" + str(index_min_2nd_f), "CONTCAR")
        path_save_f = os.path.join(Path_job, "CONTCAR_E_" + format(nif_scr + 1, '.0f') + "_" +
                                   format(relative_energy_i_f * 1000, '.0f') + "_meV")
        gen(path_log, "Structure of sorting " + format(nif_scr + 1, '.0f') +
            " (" + num_printer(index_min_f[0] + 1, np.ceil(np.log10(NSW_MC_G)) + 1, 0, 0) + "/" +
            num_printer(index_min_f[1] + 1, np.ceil(np.log10(SCQU1)) + 1, 0, 0) + ")" +
            ", relative energy: " + num_printer(relative_energy_i_f, 8, 3, 0) +
            " eV", True, True, True)
        shutil.copy(path_index_f, path_save_f)
        energy_screen_mat_f[index_min_f] = np.inf

    return energy_screened_f, atom_doped_mat_screened_f


def cal_mcce(path_log, path_tmp, atom_doped_screened_mat_f, struc_origin_f):
    gen(path_log, "", True, True, True)
    path_mcce = os.path.join(path_tmp, "temp_MCCE")
    if not os.path.exists(path_mcce):
        os.mkdir(path_mcce)

    num_group_f = atom_doped_screened_mat_f.shape[0]
    num_iteration_f = int(NSW_MCCE)
    if num_iteration_f == 0:
        return

    num_occ_f = int(FASEQ_array.shape[0] - np.sum(atom_doped_screened_mat_f[0, :] == "Va"))
    hash_mat_f = np.zeros((num_group_f, num_iteration_f), dtype=object)
    energy_mat_f = np.zeros((num_group_f, num_iteration_f))
    thermal_vf = np.zeros((num_group_f * num_iteration_f, 6))  # [all image; E, E+ZPE, U, S, G, probability]
    conf_entropy_f = np.zeros(num_group_f * num_iteration_f)
    nif_seq = 0
    nif_seq_save_mat = np.zeros((num_group_f * num_iteration_f, 2))
    gen(path_log, "****************************************************************************************************"
                  "*******************************************************************************************",
        True, True, True)
    for nif_struc in range(num_group_f):  # each group of sampling
        path_struc_opt = os.path.join(path_mcce, "Structure_" + format(nif_struc + 1, '.0f') + "_optimization")
        if not os.path.exists(path_struc_opt):
            os.mkdir(path_struc_opt)
        path_struc_phonon = os.path.join(path_mcce, "Structure_" + format(nif_struc + 1, '.0f') + "_phonon")
        if not os.path.exists(path_struc_phonon):
            os.mkdir(path_struc_phonon)

        gen(path_log, "", True, True, True)

        if num_iteration_f > 1:
            gen(path_log, "Start the configuration entropy using Monte Carlo method - Structure " +
                format(nif_struc + 1, '.0f') + " / " + format(num_group_f, '.0f') +
                "\n                                  Iteration           " +
                "Energy (eV/atoms)            dE (eV)         Acceptance rate            State\n",
                True, True, True)

        structure_save_f = {}

        # initialization
        path_res_sub = os.path.join(path_struc_opt, "tmp_" + format(nif_struc + 1, '.0f') + "_1")
        if not os.path.exists(path_res_sub):
            os.mkdir(path_res_sub)
        atom_doped_mat_f = atom_doped_screened_mat_f[nif_struc, :]
        structure_f = replace_atom(struc_origin_f, atom_doped_mat_f, FASEQ_array_index)
        hash_mat_f[nif_struc, 0] = get_structure_hash(structure_f)

        atom_name_mc_f, atom_num_mc_f = wr_atom_list(structure_f.atomic_numbers)
        relax_atom_f = np.zeros(structure_f.frac_coords.shape) + 1
        if "selective_dynamics" in structure_f.site_properties:
            relax_atom_f = np.array(structure_f.site_properties["selective_dynamics"])
        wr_contcar(path_res_sub, "POSCAR", Title, structure_f.lattice.matrix, structure_f.frac_coords,
                   atom_name_mc_f, atom_num_mc_f, relax_atom_f, np.zeros((structure_f.frac_coords.shape[0], 3)))
        energy_system_f, struc_opted_f = (
            cal_opt(Model_CHGNET, structure_f, path_res_sub, IBRION, EDIFFG3, NSW_OPT3, ISIF3, OFI_OPT3,
                    True, True, False, False, False))
        energy_mat_f[nif_struc, 0] = energy_system_f / num_occ_f
        structure_save_f[0] = struc_opted_f
        if num_iteration_f > 1:
            gen(path_log, num_printer(1, 18, 0, 0) +
                num_printer(energy_mat_f[nif_struc, 0], 24, 6, 0), True, True, True)

        structure_find_index_f = 0
        for nif_step in range(1, num_iteration_f):
            atom_doped_mat_old_f = atom_doped_mat_f.copy()
            atom_doped_mat_f = select_index(atom_doped_mat_f)
            structure_f = replace_atom(struc_origin_f, atom_doped_mat_f, FASEQ_array_index)
            hash_v = get_structure_hash(structure_f)
            if np.sum(hash_v == hash_mat_f) != 0:  # has been calculated
                continue
            else:
                hash_mat_f[nif_struc, nif_step] = hash_v
                path_res_sub = os.path.join(path_struc_opt, "tmp_" +
                                            format(nif_struc + 1, '.0f') + "_" + format(nif_step + 1, '.0f'))
                if not os.path.exists(path_res_sub):
                    os.mkdir(path_res_sub)
                # calculate the energy
                atom_name_mc_f, atom_num_mc_f = wr_atom_list(structure_f.atomic_numbers)
                relax_atom_f = np.zeros(structure_f.frac_coords.shape) + 1
                if "selective_dynamics" in structure_f.site_properties:
                    relax_atom_f = np.array(structure_f.site_properties["selective_dynamics"])
                wr_contcar(path_res_sub, "POSCAR", Title, structure_f.lattice.matrix, structure_f.frac_coords,
                           atom_name_mc_f, atom_num_mc_f, relax_atom_f, np.zeros((structure_f.frac_coords.shape[0], 3)))
                energy_system_f, struc_opted_f = (
                    cal_opt(Model_CHGNET, structure_f, path_res_sub, IBRION, EDIFFG3, NSW_OPT3, ISIF3, OFI_OPT3,
                            True, True, False, False, False))
                energy_mat_f[nif_struc, nif_step] = energy_system_f / num_occ_f
                structure_save_f[nif_step] = struc_opted_f

                delta_e_f = energy_mat_f[nif_struc, nif_step] - energy_mat_f[nif_struc, structure_find_index_f]
                delta_e_joule = delta_e_f * num_occ_f * q_cons
                acceptance_rate_f = min([1, np.exp(-delta_e_joule / kB_cons / TEMP_MC)])
                if acceptance_rate_f >= np.random.rand():
                    state_mc_f = "Acceptance"
                    structure_find_index_f = nif_step
                else:
                    state_mc_f = "*Rejection"
                    atom_doped_mat_f = atom_doped_mat_old_f

                gen(path_log,
                    num_printer(nif_step + 1, 18, 0, 0) +
                    num_printer(energy_mat_f[nif_struc, nif_step], 24, 6, 0) +
                    num_printer(delta_e_f * num_occ_f, 23, 3, 0) +
                    num_printer(acceptance_rate_f * 100, 20, 2, 0) + "%              " +
                    state_mc_f, True, True, True)

        gen(path_log, "Statistic thermodynamic properties of Structure " +
            format(nif_struc + 1, '.0f') + " / " + format(num_group_f, '.0f') +
            ":\n                                   "
            "Steps       Progress      E_elec        U          S_vib        G       |  "
            "<E_elec>        <U>         <S_vib>        S_conf         <S>         "
            "<G_tot>   (eV/atoms) / (meV/atoms/K)",
            True, True, True)
        energy_get_index_f = np.where(energy_mat_f[nif_struc, :] - np.min(energy_mat_f) <= E_MCCE / num_occ_f)[0]
        num_phonon_cal_f = energy_get_index_f.shape[0]
        min_convergence_num_f = int(max([2, num_phonon_cal_f / 10]))

        for nif_seq_image, nif_index in enumerate(energy_get_index_f.astype(int)):
            path_res_sub = os.path.join(path_struc_phonon, "tmp_" +
                                        format(nif_struc + 1, '.0f') + "_" + format(nif_index + 1, '.0f'))
            if not os.path.exists(path_res_sub):
                os.mkdir(path_res_sub)

            thermal_vf[nif_seq, : 3] = energy_mat_f[nif_struc, nif_index]  # potential energy
            gen(path_log, num_printer(nif_seq_image + 1, 10, 0, 0) + "/" +
                num_printer(energy_get_index_f.shape[0], 7, 0, 0) + "     ",
                True, True, False)
            _, _, _, _, _, vib_zpe, vib_internal_energy, vib_entropy = (
                cal_phonon(path_res_sub, Model_CHGNET, structure_save_f[nif_index], TEMP_CE,
                           True, False,
                           False, True, False))

            thermal_vf[nif_seq, 1: 3] += vib_zpe / num_occ_f  # add zero-point energy
            thermal_vf[nif_seq, 2] += vib_internal_energy / num_occ_f  # add internal energy
            thermal_vf[nif_seq, 3] = vib_entropy * 1000 / num_occ_f  # entropy in meV/atoms/K
            thermal_vf[nif_seq, 4] = thermal_vf[nif_seq, 2] - TEMP_CE * vib_entropy / num_occ_f  # add Gibbs energy
            thermal_vf[: nif_seq + 1, 5] = (np.exp(-(thermal_vf[: nif_seq + 1, 4] - np.min(thermal_vf[0, 4])) *
                                                   num_occ_f * q_cons / (kB_cons * TEMP_CE)))
            thermal_vf[: nif_seq + 1, 5] /= np.sum(thermal_vf[: nif_seq + 1, 5])

            thermal_mean_vf = np.sum(thermal_vf[:, 5: 6] * thermal_vf[:, : 5], axis=0)
            conf_entropy_f[nif_seq] = np.sum(thermal_vf[: nif_seq + 1, 5] * np.log(thermal_vf[: nif_seq + 1, 5]))
            conf_entropy_f[nif_seq] *= -kB_cons / q_cons / num_occ_f * 1000  # in meV/atoms/K

            gen(path_log,
                num_printer(thermal_vf[nif_seq, 0], 11, 4, 0) +
                num_printer(thermal_vf[nif_seq, 2], 12, 4, 0) +
                num_printer(thermal_vf[nif_seq, 3], 12, 4, 0) +
                num_printer(thermal_vf[nif_seq, 4], 12, 4, 0) + "    |" +
                num_printer(thermal_mean_vf[0], 10, 6, 0) +
                num_printer(thermal_mean_vf[2], 14, 6, 0) +
                num_printer(thermal_mean_vf[3], 14, 6, 0) +
                num_printer(conf_entropy_f[nif_seq], 14, 6, 0) +
                num_printer(thermal_mean_vf[3] + conf_entropy_f[nif_seq], 14, 6, 0) +
                num_printer(thermal_mean_vf[4] - conf_entropy_f[nif_seq] / 1000 * TEMP_CE, 14, 6, 0),
                True, False, True)

            if nif_seq_image > min_convergence_num_f:
                steps_f = np.arange(0, min_convergence_num_f + 1, min_convergence_num_f)
                steps_f_avg = np.average(steps_f)
                slope_conf_entropy = (
                    np.sum((steps_f - steps_f_avg) * (conf_entropy_f[-min_convergence_num_f:] -
                                                      np.average(conf_entropy_f[-min_convergence_num_f:]))) /
                    np.sum((steps_f - steps_f_avg) ** 2))
                if slope_conf_entropy <= CVG_S_CE:
                    gen(path_log, "Reaching convergence conditions.", True, True, True)
                    continue

            nif_seq_save_mat[nif_seq, :] = np.array([nif_struc, nif_seq_image])
            nif_seq += 1

    gen(path_log, "", True, True, True)
    gibbs_energy_array, gibbs_energy_array_min = thermal_vf[:, 4], np.min(thermal_vf[:, 4])
    for nif_struc in range(int(SCQU2)):
        index_f = np.argmin(gibbs_energy_array)
        index_mat_f = nif_seq_save_mat[index_f, :]
        relative_energy_i_f = (np.min(gibbs_energy_array) - gibbs_energy_array_min) * num_occ_f

        path_index_f = os.path.join(path_mcce, "Structure_" +
                                    format(index_mat_f[0] + 1, '.0f') + "_optimization", "tmp_" +
                                    format(index_mat_f[0] + 1, '.0f') + "_" +
                                    format(index_mat_f[1] + 1, '.0f'), "CONTCAR")
        path_save_f = os.path.join(Path_job, "CONTCAR_G_" + format(nif_struc + 1, '.0f') + "_" +
                                   format(relative_energy_i_f * 1000, '.0f') + "_meV")
        gen(path_log, "Structure of sorting " + format(nif_struc + 1, '.0f') +
            " (" + num_printer(index_mat_f[0] + 1, np.ceil(np.log10(SCQU2)) + 1, 0, 0) + "/" +
            num_printer(index_mat_f[1] + 1, np.ceil(np.log10(NSW_MCCE)) + 1, 0, 0) + ")" +
            ", relative free energy: " + num_printer(relative_energy_i_f, 8, 3, 0) +
            " eV, probability: " + num_printer(thermal_vf[index_f, 5] * 100, 8, 2, 0) + "%",
            True, True, True)
        shutil.copy(path_index_f, path_save_f)
        gibbs_energy_array[index_f] = np.inf



# Start point
if __name__ == "__main__":
    st_time = time.time()
    st_str = ("Modify by Zhong-Lun Li Sep. 16 2025\n# Department of Chemical Engineering, "
              "National Taiwan University of Science and Technology, Taipei 106, Taiwan")

    Path_job = "."
    Path_out = os.path.join(Path_job, "Output_details")
    with open(Path_out, 'w') as nf_r:
        nf_r.write("# " + st_str + "\n")
    Path_log = os.path.join(Path_job, "CHGNet_results.log")
    with open(Path_log, 'w') as nf_r:
        nf_r.write("# " + st_str + "\n\n")
    gen(Path_log, "Loading package and reading input files...", True, True, True)

    Path_pot = os.path.join(Path_job, "Fine_Tune_Model.tar")
    if os.path.exists(Path_pot):
        checkpoint = torch.load(Path_pot, map_location="cpu")
        Model_CHGNET = CHGNet.from_dict(checkpoint["model"])
        gen(Path_log, 'Read the "Fine_Tune_Model.tar" archive '
                      'and continue running using the previously trained model...\n',
            True, True, True)
    else:
        from chgnet.model import CHGNet
        Model_CHGNET = CHGNet.load()
        gen(Path_log, 'Unable to read "Fine_Tune_Model.tar" file, '
                      'starting running using pre-trained model...\n', True, True, True)

    kB_cons = 1.38064852E-23  # in J/K
    q_cons = 1.602176634E-19  # in J/eV
    NA_cons = 6.02214076E23  # in 1/mol
    AMU_cons = 1.660539040E-27  # in kg/amu
    c_cons = 299792458  # in m/s
    h_cons = 6.626069934E-34  # in J*s
    kcal_to_J_cons = 4184

    IBRION = -1
    PBC_A = True
    PBC_B = True
    PBC_C = True

    EDIFFG = 1E-2
    NSW_OPT = 500
    OFI_OPT = 1
    ISIF = True

    POTIM_PH = 0.01
    CUT_WN = 50
    TEMP_STA = 298
    GAU_FWHM = 10

    POTIM = 1
    NSW_MD = 100
    OFI_MD = 1
    ENSB = "nvt"
    THEMT = "Berendsen_inhomogeneous"
    TEMP_B = 298  # in K
    TEMP_E = 298  # in K
    TAU_T = 100
    TAU_P = 1000
    PRES = 1E-4  # in GPa
    BULK_M = 2  # in GPa

    FASEQ = []
    NAME_OCC = []
    NUM_OCC = []
    NSW_MC_G = 1
    NSW_MC_S = 1E3
    EDIFFG0 = 1E-5
    TEMP_MC = 298
    SCQU1 = 10
    EDIFFG1 = 0.05
    NSW_OPT1 = 50
    OFI_OPT1 = 1
    ISIF1 = 3
    SCQU2 = 3
    EDIFFG2 = 0.01
    NSW_OPT2 = 500
    OFI_OPT2 = 1
    ISIF2 = 3
    NSW_MCCE = 0
    E_MCCE = 3
    EDIFFG3 = 0.01
    NSW_OPT3 = 100
    OFI_OPT3 = 1
    ISIF3 = 3
    POTIM_CE = 0.01
    CUT_WNCE = 50
    TEMP_CE = 298
    CVG_S_CE = 1E-6

    if os.path.exists(os.path.join(Path_job, "Input_CHGNet")):
        rd_cncar(os.path.join(Path_job, "Input_CHGNet"))
    else:
        wr_cncar()

    if (ENSB == "npt") & (THEMT == "Berendsen"):
        THEMT = "Berendsen_inhomogeneous"

    Path_pos = os.path.join(Path_job, "POSCAR")
    if not os.path.exists(Path_pos):
        gen(Path_log, 'The "POSCAR" does not exist '
                      'and the program terminates abnormally.\n', True, True, True)
        exit(0)
    Title, Atom_type, Atom_num, LC, Coord, Coord_relax, STRUCture = rd_pos(Path_pos)
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.filterwarnings("ignore", message="logm result may be inaccurate")

    if IBRION == -1:
        gen(Path_log, 'Reading completed, calculation method is "SIngle point calculation".\n' 
                      '                           '
                      'Steps      Volume             E0               dE             '
                      'Force_max         Magnetic moment\n                                   '
                      '(Angstrom^3)      (eV/atoms)       (eV/atoms)       (eV/Angstrom)          (mu_B)\n', True, True, True)
        cal_sp(Model_CHGNET, 0, 0)

    elif IBRION == 0:
        gen(Path_log, 'Reading completed, calculation method is "Molecular Dynamics".\n'
                      '                              '
                      'Time     Temperature     Volume          E0           dE           E0_K         dE_K          '
                      'E0_P         dE_P       Force_max      Magnetic moment\n'
                      '                              (ps)         (K)      (Angstrom^3)   '
                      '(eV/atoms)   (eV/atoms)    (eV/atoms)   (eV/atoms)    (eV/atoms)   '
                      '(eV/atoms)  (eV/Angstrom)       (mu_B)\n', True, True, True)

        warnings.filterwarnings("ignore", module="pymatgen")
        warnings.filterwarnings("ignore", module="ase")

        Process_MD = Process(target=cal_md_sub,
                             args=(Model_CHGNET, Path_job, NSW_MD, STRUCture, ENSB, THEMT, TEMP_B, TEMP_E,
                                   POTIM, OFI_MD, TAU_T, TAU_P, PRES, BULK_M))
        Process_Record = Process(target=cal_md_record_hold,
                                 args=(Path_job, Path_log, Path_out,
                                       NSW_MD, ENSB, Title, Atom_type, Atom_num, OFI_MD, POTIM, Coord_relax, 10, 6))

        Process_MD.start()
        Process_Record.start()

        try:
            Process_MD.join()
            Process_Record.join()
        except KeyboardInterrupt:
            Process_MD.terminate()
            Process_Record.terminate()
            Process_MD.join()
            Process_Record.join()

    elif IBRION == 5:
        (Num_real_large_freq, Num_real_low_freq, Num_image_freq,
         Vib_freq, Vib_mode, Vib_ZPE, Vib_Internal_energy, Vib_Entropy) = (
            cal_phonon(Path_job, Model_CHGNET, STRUCture, TEMP_STA,
                       True, True,
                       True, True, True))

    elif IBRION % 10 == 5:
        if IBRION // 10 == 1:
            IBRION = "LBFGS"
        elif IBRION // 10 == 3:
            IBRION = "FIRE"

        _, STRUCture = (
            cal_opt(Model_CHGNET, STRUCture, Path_job, IBRION, EDIFFG, NSW_OPT, ISIF, OFI_OPT,
                    True, True, True, True, True))

        gen(Path_log, "", True, True, True)
        gen(Path_log, "Structural optimization is completed and phonon calculation is performed...",
            True, True, True)
        (Num_real_large_freq, Num_real_low_freq, Num_image_freq, Vib_freq,
         Vib_mode, Vib_ZPE, Vib_Internal_energy, Vib_Entropy) = (
            cal_phonon(Path_job, Model_CHGNET, STRUCture, TEMP_STA,
                       True, True,
                       True, True, True))

    elif IBRION // 10 == 0:
        if IBRION % 10 == 1:
            IBRION = "LBFGS"
        elif IBRION % 10 == 3:
            IBRION = "FIRE"

        _, _ = cal_opt(Model_CHGNET, STRUCture, Path_job, IBRION, EDIFFG, NSW_OPT, ISIF, OFI_OPT,
                       True, True, True, True, True)

    elif IBRION // 10 == 1:
        if IBRION % 10 == 1:
            IBRION = "LBFGS"
        elif IBRION % 10 == 3:
            IBRION = "FIRE"

        gen(Path_log, 'Reading completed, calculation method is "Sampling Optimization".',
            True, True, True)

        FASEQ_array = np.zeros(STRUCture.num_sites)
        for ni in range(len(FASEQ)):
            data_i = FASEQ[ni].split(":")
            if len(data_i) == 3:
                FASEQ_array[int(data_i[0]): int(data_i[2]) + 1: int(data_i[1])] = True
            elif len(data_i) == 2:
                FASEQ_array[int(data_i[0]): int(data_i[1]) + 1] = True
            elif len(data_i) == 1:
                FASEQ_array[int(data_i[0])] = True
        FASEQ_array_index = np.where(FASEQ_array)[0]

        if len(NAME_OCC) != len(NUM_OCC):
            gen(Path_log, 'Error! The lengths of the atom type label "' + " ".join(NAME_OCC) +
                          '" and the value label "' + " ".join(NUM_OCC) + '" do not match! Program aborted.\n',
                True, True, True)
            exit(0)

        NSW_MC_G_dight = np.ceil(np.log10(NSW_MC_G) + 1)
        NSW_MC_S_dight = np.ceil(np.log10(NSW_MC_S) + 1)
        STRUCture_site_prop = STRUCture.site_properties
        Path_tmp = os.path.join(Path_job, "CHGNet_tmp")
        if not os.path.exists(Path_tmp):
            os.mkdir(Path_tmp)

        # 1st screening
        Num_occ, Energy_save_mat, STRUCture_screen, Atom_doped_mat = cal_mc(Path_log, STRUCture)
        # 2nd screening
        Energy_screened, Atom_doped_screened_mat = cal_mc_opt(Path_log, Path_tmp)
        # Configuration calculation using Monte Carlo method
        cal_mcce(Path_log, Path_tmp, Atom_doped_screened_mat, STRUCture)


    gen(Path_log, "", True, True, True)
    gen(Path_log, "Program terminated! This took " +
        num_printer((time.time() - st_time) / 3600, 8, 2, 0) + " hours.",
        True, True, True)

