# Modify by Zhong-Lun Li Dec. 17 2025,
# Department of Chemical Engineering, National Taiwan University of Science and Technology, Taipei 106, Taiwan

# The CHGNet model comes from this study:
# Deng, B., Zhong, P., Jun, K. et al. CHGNet as a pretrained universal neural network potential for
# charge-informed atomistic modelling. Nat Mach Intell 5, 1031–1041 (2023). https://doi.org/10.1038/s42256-023-00716-3

import os, time, io, sys, warnings, torch, shutil, hashlib, json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from multiprocessing import Process
from functools import partial

from ase import io as ase_io
from ase.gui.surfaceslab import structures
from ase.lattice.bravais import Lattice
from ase.filters import FrechetCellFilter

from chgnet.model.model import CHGNet
from chgnet.model.dynamics import StructOptimizer
from matplotlib.lines import lineStyles
import matplotlib.colors as mcolors
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.core import Structure, Lattice
from pymatgen.core.sites import DummySpecies
from pymatgen.core.surface import SlabGenerator
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sympy.core.multidimensional import structure_copy


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

    atoms_f = ase_io.read(path_f)
    atoms_f.pbc = [PBC_A, PBC_B, PBC_C]
    struc_f = AseAtomsAdaptor.get_structure(atoms_f)
    if "selective_dynamics" in struc_f.site_properties:
        struc_f.add_site_property("selective_dynamics", coord_relax == 1)

    return d[0], atom_type, atom_num, lc, coord, coord_relax, struc_f


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
        elif data_nif[0] == "XRD_FWHM":
            global XRD_FWHM
            XRD_FWHM = float(data_nif[-1])
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
        elif data_nif[0] == "DOF_A":
            global DOF_A
            if int(data_nif[-1]) == 0:
                DOF_A = False
            elif int(data_nif[-1]) == 1:
                DOF_A = True
        elif data_nif[0] == "DOF_B":
            global DOF_B
            if int(data_nif[-1]) == 0:
                DOF_B = False
            elif int(data_nif[-1]) == 1:
                DOF_B = True
        elif data_nif[0] == "DOF_C":
            global DOF_C
            if int(data_nif[-1]) == 0:
                DOF_C = False
            elif int(data_nif[-1]) == 1:
                DOF_C = True
        elif data_nif[0] == "DOF_BC":
            global DOF_BC
            if int(data_nif[-1]) == 0:
                DOF_BC = False
            elif int(data_nif[-1]) == 1:
                DOF_BC = True
        elif data_nif[0] == "DOF_AC":
            global DOF_AC
            if int(data_nif[-1]) == 0:
                DOF_AC = False
            elif int(data_nif[-1]) == 1:
                DOF_AC = True
        elif data_nif[0] == "DOF_AB":
            global DOF_AB
            if int(data_nif[-1]) == 0:
                DOF_AB = False
            elif int(data_nif[-1]) == 1:
                DOF_AB = True
        elif data_nif[0] == "NSW_SOLC":
            global NSW_SOLC
            NSW_SOLC = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "RAT_SOLC":
            global RAT_SOLC
            RAT_SOLC = float(data_nif[-1])
        elif data_nif[0] == "DOF_SO1":
            global DOF_SO1
            for nif_option in range(len(data_nif) - 2):
                DOF_SO1.append(data_nif[2 + nif_option])
        elif data_nif[0] == "DOF_SO2":
            global DOF_SO2
            for nif_option in range(len(data_nif) - 2):
                DOF_SO2.append(data_nif[2 + nif_option])
        elif data_nif[0] == "DOF_SOL1":
            global DOF_SOL1
            DOF_SOL1 = float(data_nif[-1])
        elif data_nif[0] == "DOF_SOH1":
            global DOF_SOH1
            DOF_SOH1 = float(data_nif[-1])
        elif data_nif[0] == "DOF_SON1":
            global DOF_SON1
            DOF_SON1 = int(float(data_nif[-1]))
        elif data_nif[0] == "DOF_SOL2":
            global DOF_SOL2
            DOF_SOL2 = float(data_nif[-1])
        elif data_nif[0] == "DOF_SOH2":
            global DOF_SOH2
            DOF_SOH2 = float(data_nif[-1])
        elif data_nif[0] == "DOF_SON2":
            global DOF_SON2
            DOF_SON2 = int(float(data_nif[-1]))
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
        elif "FASEQ_" in data_nif[0]:
            if data_nif[0][:6] == "FASEQ_":
                global FASEQ
                key_f = data_nif[0][6:]
                FASEQ[key_f] = data_nif[2:]
        elif "NAME_OCC_" in data_nif[0]:
            if data_nif[0][:9] == "NAME_OCC_":
                global NAME_OCC_dict
                key_f = data_nif[0][9:]
                NAME_OCC_dict[key_f] = data_nif[2:]
        elif "NUM_OCC_" in data_nif[0]:
            if data_nif[0][:8] == "NUM_OCC_":
                global NUM_OCC_dict
                key_f = data_nif[0][8:]
                NUM_OCC_dict[key_f] = data_nif[2:]
        elif data_nif[0] == "MIN_R_MC":
            global MIN_R_MC
            MIN_R_MC = float(data_nif[-1])
        elif data_nif[0] == "NSW_MC_G":
            global NSW_MC_G
            NSW_MC_G = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "NSW_MC_S":
            global NSW_MC_S
            NSW_MC_S = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "EDIFFG0":
            global EDIFFG0
            EDIFFG0 = float(data_nif[-1])
        elif data_nif[0] == "TEMP_MCI":
            global TEMP_MCI
            TEMP_MCI = float(data_nif[-1])
        elif data_nif[0] == "TEMP_MCF":
            global TEMP_MCF
            TEMP_MCF = float(data_nif[-1])
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
        elif data_nif[0] == "NUM_CONF":
            global NUM_CONF
            NUM_CONF = int(float(data_nif[-1]))
        elif data_nif[0] == "E_SACONF":
            global E_SACONF
            E_SACONF = float(data_nif[-1])

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
        elif data_nif[0] == "EDIFFG_M":
            global EDIFFG_M
            EDIFFG_M = abs(float(data_nif[-1]))
        elif data_nif[0] == "NSW_MP":
            global NSW_MP
            NSW_MP = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "OFI_MP":
            global OFI_MP
            OFI_MP = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "RATIO_MP":
            global RATIO_MP
            RATIO_MP = float(data_nif[-1])
        elif data_nif[0] == "EDIFFG_T":
            global EDIFFG_T
            EDIFFG_T = float(data_nif[-1])
        elif data_nif[0] == "POTIM_T":
            global POTIM_T
            POTIM_T = float(data_nif[-1])
        elif data_nif[0] == "NSW_T":
            global NSW_T
            NSW_T = int(float(data_nif[-1]))
        elif data_nif[0] == "IMAGES":
            global IMAGES
            IMAGES = int(float(data_nif[-1]))
        elif data_nif[0] == "SPRING_S":
            global SPRING_S
            SPRING_S = float(data_nif[-1])
        elif data_nif[0] == "SPRING_L":
            global SPRING_L
            SPRING_L = float(data_nif[-1])
        elif data_nif[0] == "MIL_IND":
            global MIL_IND
            MIL_IND = np.array([float(data_nif[-3]), float(data_nif[-2]), float(data_nif[-1])])
        elif data_nif[0] == "NUM_TB":
            global NUM_TB
            NUM_TB = int(float(data_nif[-1]))
        elif data_nif[0] == "NUM_FB":
            global NUM_FB
            NUM_FB = int(float(data_nif[-1]))
        elif data_nif[0] == "H_VAC":
            global H_VAC
            H_VAC = float(data_nif[-1])
        elif data_nif[0] == "EDIFFG_I":
            global EDIFFG_I
            EDIFFG_I = abs(float(data_nif[-1]))
        elif data_nif[0] == "NSW_I":
            global NSW_I
            NSW_I = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "OFI_I":
            global OFI_I
            OFI_I = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "NUM_CS":
            global NUM_CS
            NUM_CS = int(float(data_nif[-1]))
        elif data_nif[0] == "EDIFFG_F":
            global EDIFFG_F
            EDIFFG_F = abs(float(data_nif[-1]))
        elif data_nif[0] == "NSW_F":
            global NSW_F
            NSW_F = max([1, int(float(data_nif[-1]))])
        elif data_nif[0] == "OFI_F":
            global OFI_F
            OFI_F = max([1, int(float(data_nif[-1]))])


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
                                    "  # single point (-1); molecular dynamic (0); optimize using RMM-DIIS (1) or FIRE (3); \n"
                                    "  # phonon calculation (5); optimize with phonon calculation "
                                    "using RMM-DIIS (15) or FIRE (35);\n"
                                    "  # sampling optimization using RMM-DIIS (11) or FIRE (13)\n"
                                    "  # scan under constrained optimization using RMM-DIIS (21) or FIRE (23)\n"
                                    "  # mechanical property calculation using RMM-DIIS (71) or FIRE (73)\n"
                                    "  # NEB calculation using steepest descent (80) or FIRE (83)\n"
                                    "  # CI-NEB calculation using steepest descent (90) or FIRE (93)\n"
                                    "  # Cleave and calculate the surface energy using RMM-DIIS (101) or FIRE (103)\n"
                                    "  PBC_A    =   1       # [1]    # considering the periodicity in the a vector (1) or not (0)\n"
                                    "  PBC_B    =   1       # [1]    # considering the periodicity in the b vector (1) or not (0)\n"
                                    "  PBC_C    =   1       # [1]    # considering the periodicity in the c vector (1) or not (0)\n"
                                    "  XRD_FWHM =  0.2      # [0.2]  # full width at half maximum (FWHM) "
                                    "of the Gaussian smoothing (in 2-theta)\n"
                                    "\n\n# For optimization / scan under constrained optimization\n"
                                    "#  Tags      Options   Default  Comments\n"
                                    "  EDIFFG   =  1E-2     # [0.01] # convergence condition, maximum force on atoms (in eV/Angstrom)\n"
                                    "  NSW_OPT  =  500      # [500]  # maximum number of steps for geometry optimization\n"
                                    "  OFI_OPT  =   1       # [1]    # output frame interval\n"
                                    "  ISIF     =   3       # [3]    # fix cell (2); relax cell (3)\n"
                                    "  # DOF_"": the options for constrained optimization (DOF_SO_*) "
                                    "are independent of the DOF_* selections\n"
                                    "  DOF_A    =   1       # [1]    # controls the degree of freedom of the lattice vector A, "
                                    "released (1) or fixed (0)\n"
                                    "  DOF_B    =   1       # [1]    # controls the degree of freedom of the lattice vector B, "
                                    "released (1) or fixed (0)\n"
                                    "  DOF_C    =   1       # [1]    # controls the degree of freedom of the lattice vector C, "
                                    "released (1) or fixed (0)\n"
                                    "  DOF_BC   =   1       # [1]    # controls the degree of freedom that the lattice angle between "
                                    "vectors B and C, either released (1) or fixed (0)\n"
                                    "  DOF_AC   =   1       # [1]    # controls the degree of freedom that the lattice angle between "
                                    "vectors A and C, either released (1) or fixed (0)\n"
                                    "  DOF_AB   =   1       # [1]    # controls the degree of freedom that the lattice angle between "
                                    "vectors A and B, either released (1) or fixed (0)\n"
                                    "\n# Just for scan under constrained optimization\n"
                                    "  NSW_SOLC =  100      # [100]  # initialization of the lattice shape "
                                    "without changing the fractional coordinates of atoms\n"
                                    "  RAT_SOLC =  1E-2     # [1E-2] # initialization of "
                                    "the lattice shape changes proportionally\n"
                                    "  DOF_SO1  =           # []     # the 1st constraint optimizer's lattice vectors or angles, "
                                    "you can choose multiple or leave it empty\n"
                                    "  DOF_SO2  =           # []     # the 2nd constraint optimizer's lattice vectors or angles, "
                                    "you can choose multiple or leave it empty\n"
                                    "  DOF_SOL1 =  0.8      # [0.8]  # the 1st constraint optimizer's scan lower bound ratio\n"
                                    "  DOF_SOH1 =  1.2      # [1.2]  # the 1st constraint optimizer's scan higher bound ratio\n"
                                    "  DOF_SON1 =   10      # [10]   # The 1st constraint on the number of scans of the optimizer\n"
                                    "  DOF_SOL2 =  0.8      # [0.8]  # the 2nd constraint optimizer's scan lower bound ratio\n"
                                    "  DOF_SOH2 =  1.2      # [1.2]  # the 2nd constraint optimizer's scan higher bound ratio\n"
                                    "  DOF_SON2 =   10      # [10]   # The 2nd constraint on the number of scans of the optimizer\n"
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
                                    "# FASEQ_1  = \n"
                                    '  # FASEQ_X: "X" indicates different sampling groups and starting from 1; '
                                    'the same applies to "NAME_OCC" and "NUM_OCC".\n'""
                                    "  #          The serial numbers start from 0, different serial numbers "
                                    "are separated by space and written in the same column.\n"
                                    "  #          You can use colons to represent consecutive serial numbers.\n"
                                    "The following two examples show the same representation method, \n"
                                    "  #                                         0 1 3 5 10 11 12 18 21\n"
                                    "  #                                         0 1:2:5 10:12 18 21\n"
                                    "# NAME_OCC_1 = \n"
                                    "# NUM_OCC_1 = \n"
                                    '  # NAME_OCC_X: the name of the atom to be placed, if it is vacancy, the name is "Va"\n'
                                    "  # NUM_OCC_X: the number of atoms to be placed\n"
                                    "# MIN_R_MC =           # [1/(X+1)] # minimum sampling ratio for each combination\n"
                                    "  NSW_MC_G =   1       # [1]    # number of Monte Carlo iteration groups\n"
                                    "  NSW_MC_S =  1E3      # [1E3]  # number of Monte Carlo iterations in the 1st stage of screening\n"
                                    "  EDIFFG0  =  1E-5     # [1E-5] # convergence condition of Monte Carlo method (in eV/atoms)\n"
                                    "  TEMP_MCB =  298      # [298]  # initial temperature of Monte Carlo method (in K)\n"
                                    "  TEMP_MCF =  298      # [298]  # final temperature of Monte Carlo method (in K)\n\n"
                                    "# 1st stage\n"
                                    "  SCQU1    =   10      # [10]   # screening quantity\n"
                                    "  EDIFFG1  =  5E-2     # [0.05] # convergence condition, maximum force on atoms (in eV/Angstrom)\n"
                                    "  NSW_OPT1 =   50      # [50]   # maximum number of steps for geometry optimization\n"
                                    "  OFI_OPT1 =   1       # [1]    # output frame interval\n"
                                    "  ISIF1    =   3       # [3]    # fix cell (2); relax cell (3)\n\n"
                                    "# sampling stage (only used for fast sampling, unrelated to main computation)\n"
                                    "  NUM_CONF =  100      # [100]  # number of sampling configurations\n"
                                    "  E_SACONF =   1       # [1]    # the upper limit of energy of the sampling configuration (in eV)\n\n"
                                    "# 2nd stage\n"
                                    "  SCQU2    =   3       # [3]    # screening quantity for all sampling data\n"
                                    "  EDIFFG2  =  1E-2     # [0.01] # convergence condition, maximum force on atoms (in eV/Angstrom)\n"
                                    "  NSW_OPT2 =  500      # [500]  # maximum number of steps for geometry optimization\n"
                                    "  OFI_OPT2 =   1       # [1]    # output frame interval\n"
                                    "  ISIF2    =   3       # [3]    # fix cell (2); relax cell (3)\n\n"
                                    "# 3rd stage (configuration entropy, the settings for phonon calculations are irrelevant here)\n"
                                    "  NSW_MCCE =   0       # [0]    # number of Monte Carlo iterations "
                                    "in the configuration entropy calculation\n"
                                    "  E_MCCE   =   1       # [1]    "
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
                                    "the configuration entropy slope (in meV/atoms/K/steps)\n"
                                    "\n\n# For mechanical property calculation\n"
                                    "#  Tags      Options   Default  Comments\n"
                                    "  EDIFFG_M =  1E-2     # [0.01] # convergence condition, maximum force on atoms (in eV/Angstrom)\n"
                                    "  NSW_MP   =  100      # [500]  # maximum number of steps for geometry optimization\n"
                                    "  OFI_MP   =   1       # [1]    # output frame interval\n"
                                    "  RATIO_MP =  0.01     # [0.01] # lattice transformation scaling for finite difference methods\n"
                                    "\n\n# For NEB/CI-NEB calculation\n"
                                    "#  Tags      Options   Default  Comments\n"
                                    "  EDIFFG_T =  1E-2     # [1E-2] # convergence condition, maximum force on atoms "
                                    "at each image (in eV/Angstrom)\n"
                                    "  POTIM_T  =  0.1      # [0.1]  # the maximum displacement of atomic movement "
                                    "during the steepest descent optimization process (in Angstrom)\n"
                                    "  NSW_T    =  100      # [100]  # maximum number of steps for geometry optimization\n"
                                    "# IMAGES   =           # [Auto] # the number of images to be optimized, "
                                    "excluding the initial and final states\n"
                                    "  SPRING_S =   5       # [5]    # minimum number of spring constant for elastic band, "
                                    "used for automatic adjustment (in eV/Angstrom^2)\n"
                                    "  SPRING_L =   5       # [5]    # maximum number of spring constant for elastic band, "
                                    "used for automatic adjustment (in eV/Angstrom^2)\n"
                                    "\n\n# For the surface energy calculation\n"
                                    "#  Tags      Options   Default  Comments\n"
                                    "# MIL_IND  =           # []     # Miller index, 3 integers and separated by spaces\n"
                                    "  NUM_TB   =    2      # [2]    # integer, total number of units on the surface\n"
                                    "  NUM_FB   =    1      # [1]    # integer, total number of units fixed on the surface\n"
                                    "  H_VAC    =   10      # [10]   # the thickness of vacuum layer (in Angstrom)\n"
                                    "  EDIFFG_I =  5E-1     # [0.05] # convergence condition, maximum force on atoms (in eV/Angstrom)\n"
                                    "  NSW_I    =   50      # [50]   # maximum number of steps for geometry optimization\n"
                                    "  OFI_I    =    1      # [1]    # output frame interval\n\n"
                                    "  NUM_CS   =   10      # [10]   # the number of structures after screening\n"
                                    "  EDIFFG_F =  1E-2     # [0.01] # convergence condition, maximum force on atoms (in eV/Angstrom)\n"
                                    "  NSW_F    =   500     # [500]  # maximum number of steps for geometry optimization\n"
                                    "  OFI_F    =    1      # [1]    # output frame interval\n")
    exit(0)


def num_printer(data_ori, sp, dg, tp):
    # tp: full number: 0, scientific notation: 1
    data = np.float64(data_ori)
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
        if (atom_name_f[nif] >= 0) & (atom_name_f[nif] <= 118):
            atom_name_f_str[nif] = Element.from_Z(int(atom_name_f[nif])).symbol
        else:
            atom_name_f_str[nif] = "Va"

    return atom_name_f_str, atom_num_f


def wr_contcar(path_job, filename, title, data_lc, data_coord, atom_type, atom_num, coord_relax, velocity_f):
    if velocity_f is None:
        velocity_f = np.zeros(data_coord.shape)
        velocity_f_none = True
    else:
        velocity_f_none = False
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

    if not velocity_f_none:
        for nif in range(velocity_f.shape[0]):
            for nif2 in range(3):
                st_f += num_printer(velocity_f[nif, nif2], 16, 8, 1)
            st_f += "\n"

    with open(os.path.join(path_job, filename), 'w') as nf_rf:
        nf_rf.write(st_f)


def wr_xdatcar_head(path_job, file_name_f, data_lc, title, atom_type, atom_num):
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

    with open(os.path.join(path_job, file_name_f), 'a') as nf_rf:
        nf_rf.write(st_f + "\n")


def wr_xdatcar_coord(path_job, file_name_f, step_f, data_coord):
    st_f = "Direct configuration=" + num_printer(step_f, 8, 0, 0) + "\n"

    for nif in range(data_coord.shape[0]):
        st_f += " "
        for nif2 in range(3):
            st_f += num_printer(data_coord[nif, nif2], 12, 8, 0)
        st_f += "\n"

    with open(os.path.join(path_job, file_name_f), 'a') as nf_rf:
        nf_rf.write(st_f)


def ionic_step_printer(energy_f, d_energy, atom_coord_f, force_f, stress_f, mag_f, temp_f, volume_f, lc_mat_f):
    num_atom_f = atom_coord_f.shape[0]

    if np.isscalar(force_f):
        force_f_max_sca = force_f
    else:
        force_f_max_sca = np.max(np.abs(np.linalg.norm(force_f, axis=1)))

    if isinstance(energy_f, np.ndarray):  # MD
        if energy_f.size == 3:
            logfile_str_f = (num_printer(temp_f, 12, 2, 0) +
                             num_printer(volume_f, 15, 3, 0) +
                             num_printer(energy_f[0] * num_atom_f, 16, 6, 1) +
                             num_printer(d_energy[0] * num_atom_f, 10, 2, 0) +
                             num_printer(energy_f[1] * num_atom_f, 17, 6, 1) +
                             num_printer(d_energy[1] * num_atom_f, 10, 2, 0) +
                             num_printer(energy_f[2] * num_atom_f, 17, 6, 1) +
                             num_printer(d_energy[2] * num_atom_f, 10, 2, 0) +
                             num_printer(force_f_max_sca, 13, 3, 0) +
                             num_printer(np.sum(mag_f), 17, 2, 0))
            energy_f = energy_f[0]
        else:
            logfile_str_f = (num_printer(volume_f, 15, 3, 0) +
                             num_printer(energy_f * num_atom_f, 20, 8, 1) +
                             num_printer(d_energy * num_atom_f, 13, 2, 0) +
                             num_printer(force_f_max_sca, 19, 6, 0))
            if not np.isnan(mag_f).all():
                logfile_str_f += num_printer(np.sum(mag_f), 20, 4, 0)
    else:
        logfile_str_f = (num_printer(volume_f, 15, 3, 0) +
                         num_printer(energy_f * num_atom_f, 20, 8, 1) +
                         num_printer(d_energy * num_atom_f, 13, 2, 0) +
                         num_printer(force_f_max_sca, 19, 6, 0))
        if not np.isnan(mag_f).all():
            logfile_str_f += num_printer(np.sum(mag_f), 20, 4, 0)

    outcar_str_f = ("  Energy: " + num_printer(energy_f * num_atom_f, 22, 8, 1) +
                    " eV           Volume: " + num_printer(volume_f, 16, 6, 0) +
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


def plot_f(data_xf, data_yf, legend_f, color_f, x_label_f, y_label_f, title_f, path_f, tp_f):
    if tp_f == 1:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(data_xf, data_yf, 'o-', label=legend_f, color=color_f)
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
        ax.plot(data_xf, data_yf, label=legend_f, color=color_f)
        ax.set_xlabel(x_label_f, fontsize=20, color="k", fontweight="bold")
        ax.set_ylabel(y_label_f, fontsize=20, color="k", fontweight="bold")
        plt.legend(fontsize=16)
        plt.title(title_f, fontsize=24, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=16)
        plt.savefig(path_f, dpi=200)
        plt.close()


def rm_dummy_atom(struc_f):
    struc_clear_f = struc_f.copy()
    struc_clear_f.remove_sites([nif for nif, site in enumerate(struc_f) if "X" in site.species_string])

    return struc_clear_f


def cal_sp(model_chgnet, step_f, d_energy):
    prediction = model_chgnet.predict_structure(STRUCture)
    atom_coord_f = np.zeros((len(STRUCture), 3))
    for nif in range(atom_coord_f.shape[0]):
        atom_coord_f[nif, :] = STRUCture[nif].frac_coords
    logfile_str, outcar_str = ionic_step_printer(prediction["e"], d_energy, atom_coord_f,
                                                 prediction["f"], prediction["s"] * 160.21766208, prediction["m"],
                                                 np.nan, STRUCture[nif].lattice.volume, STRUCture[nif].lattice.matrix)

    gen(Path_log, num_printer(0, 6, 0, 0) + logfile_str, True, True, True)

    with open(Path_out, 'a') as nf_rf:
        nf_rf.write("Step " + num_printer(step_f, 8, 0, 0) + "        " + outcar_str)


def cal_xrd(struc_f, path_job):
    calc_f = XRDCalculator(wavelength="CuKa")
    res_f = calc_f.get_pattern(struc_f, two_theta_range=(10, 90))
    two_theta_f = res_f.x
    inten_f = res_f.y
    pattern_f = res_f.hkls
    path_data = os.path.join(path_job, "XRD_pattern.log")
    st_f = "# " + st_str + "\n\n    2-theta (degree)    Relative intensity (%)    Miller index\n"
    gau_std_f = XRD_FWHM / (2 * np.sqrt(2 * np.log(2)))
    xrd_2_theta_f = np.arange(two_theta_f[0] - 3 * gau_std_f, two_theta_f[-1] + 3 * gau_std_f, gau_std_f / 10)
    xrd_inten_f = np.zeros(xrd_2_theta_f.shape[0])
    for nif_pattern in range(len(pattern_f)):
        st_f += (num_printer(two_theta_f[nif_pattern], 15, 4, 0) +
                 num_printer(inten_f[nif_pattern], 23, 3, 0))
        pat_partial_f = pattern_f[nif_pattern][0]["hkl"]
        if len(pat_partial_f) == 3:
            st_f += "          "
        else:
            st_f += "        "
        for nif_mi in range(len(pat_partial_f)):
            st_f += num_printer(pat_partial_f[nif_mi], 5, 0, 0)
        st_f += "\n"

        xrd_inten_f += (inten_f[nif_pattern] *
                        np.exp((xrd_2_theta_f - two_theta_f[nif_pattern]) ** 2 / (-2 * gau_std_f ** 2)))
    with open(path_data, 'w') as nf_rf:
        nf_rf.write(st_f)

    xrd_inten_f /= 100

    path_data = os.path.join(path_job, "XRD_diagram.log")
    st_f = "# " + st_str + "\n\n      2-theta (degree)    Relative intensity (%)\n"
    for nif_spectra in range(xrd_2_theta_f.shape[0]):
        st_f += (num_printer(xrd_2_theta_f[nif_spectra], 15, 4, 0) +
                 num_printer(xrd_inten_f[nif_spectra], 23, 3, 0) + "\n")
    with open(path_data, 'w') as nf_rf:
        nf_rf.write(st_f)

    plot_f(xrd_2_theta_f, xrd_inten_f, "XRD", "b", "2-theta (degree)",
           "Relative intensity", "XRD - Standard deviation: " + format(gau_std_f, '.2f') + " degree",
           os.path.join(path_job, "XRD_diagram.png"), 0)


def cal_md_sub(model_chgnet, path_job, nsw_md, structure, ensb, themt, temp_b, temp_e, potim, ofi_md, tau_t, tau_p,
               pres, bulk_m):
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
        lc_array_f = traj.cell.cellpar()
        lc_mat_f = Lattice.from_parameters(lc_array_f[0], lc_array_f[1], lc_array_f[2],
                                           lc_array_f[3], lc_array_f[4], lc_array_f[5]).matrix
        wr_xdatcar_head(path_job, "Trajectory_VASP", lc_mat_f, title, atom_type, atom_num)


def cal_md_record(path_job, path_log, path_out, energy_hold, st_frame, ed_frame, title, atom_type, atom_num,
                  ensb, ofi_md, potim, coord_relax, plt_data_f, plt_boolean):
    for nif in range(st_frame, ed_frame):
        from ase.io import read
        nif_data = read(os.path.join(path_job, "CHGNet_md.traj"), index=nif)
        lc_array_f = nif_data.cell.cellpar()
        lc_mat_f = Lattice.from_parameters(lc_array_f[0], lc_array_f[1], lc_array_f[2],
                                           lc_array_f[3], lc_array_f[4], lc_array_f[5]).matrix
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

        gen(path_log, num_printer(nif * ofi_md * potim / 1000, 11, 3, 0) +
            logfile_str, True, True, True)

        with open(path_out, 'a') as nf_rf:
            nf_rf.write("\nTime " + num_printer(nif * ofi_md * potim / 1000, 8, 3, 0) +
                        " ps      " + outcar_str)

        if ensb == "npt":  # relax cell
            wr_xdatcar_head(path_job, "Trajectory_VASP", lc_mat_f, title, atom_type, atom_num)
        wr_xdatcar_coord(path_job, "Trajectory_VASP", nif * ofi_md, coord_fraction_f)

        plt_data_f[nif, :] = np.array([nif * ofi_md * potim / 1000, nif_data.get_temperature(), nif_data.get_volume(),
                                       energy_md[0], energy_md[-1], -np.average(np.diag(stress_mat_f)),
                                       np.sum(nif_data.get_magnetic_moments())])

    if plt_boolean:
        # plot
        plot_f(plt_data_f[:ed_frame, 0], plt_data_f[:ed_frame, 1],
               "Temperature", "r", "Simulation time (ps)", "Temperature (K)",
               "Temperature vs. Time", os.path.join(path_job, "_CHGNet_Temperature.png"), 0)
        plot_f(plt_data_f[:ed_frame, 0], plt_data_f[:ed_frame, 2],
               "Volume", "k", "Simulation time (ps)", "Volume (Angstrom^3)",
               "Volume vs. Time", os.path.join(path_job, "_CHGNet_Volume.png"), 0)
        plot_f(plt_data_f[:ed_frame, 0], (plt_data_f[:ed_frame, 3] - plt_data_f[0, 3]) * np.sum(atom_num),
               "Total energy difference (eV)", "k",
               "Simulation time (ps)", "Total energy difference (eV)",
               "Total energy vs. Time", os.path.join(path_job, "_CHGNet_Total energy.png"), 0)
        plot_f(plt_data_f[:ed_frame, 0], (plt_data_f[:ed_frame, 4] - plt_data_f[0, 4]) * np.sum(atom_num),
               "Potential energy difference", "b",
               "Simulation time (ps)", "Potential energy difference (eV)",
               "Potential energy vs. Time", os.path.join(path_job, "_CHGNet_Potential energy.png"), 0)
        plot_f(plt_data_f[:ed_frame, 0], (plt_data_f[:ed_frame, 3] - plt_data_f[0, 3] -
                                          plt_data_f[:ed_frame, 4] + plt_data_f[0, 4]) * np.sum(atom_num),
               "Kinetic energy difference", "r",
               "Simulation time (ps)", "Kinetic energy difference (eV)",
               "Kinetic energy vs. Time", os.path.join(path_job, "_CHGNet_Kinetic energy.png"), 0)
        plot_f(plt_data_f[:ed_frame, 0], plt_data_f[:ed_frame, 5] * 1E4, "Hydrostatic stress", "k",
               "Simulation time (ps)", "Hydrostatic stress (bar)", "Stress vs. Time",
               os.path.join(path_job, "_CHGNet_Hydrostatic stress.png"), 0)
        plot_f(plt_data_f[:ed_frame, 0], plt_data_f[:ed_frame, 6], "Magnetic moment", "k",
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


def cal_opt(model_chgnet, structure_f, path_job, ibrion, ediffg, nsw, isif, ofi, dof_array,
            output_contcar, output_log, output_outcar, output_xdatcar, output_diagram):
    if output_log:
        path_log = os.path.join(path_job, "CHGNet_results.log")
        with open(path_log, 'a') as nf_rf:
            nf_rf.write("# " + st_str + "\n\n")
        gen(path_log, 'Reading completed, calculation method is "Geometry Optimization".\n'
                      '                           '
                      'Steps      Volume             E0               dE             '
                      'Force_max         Magnetic moment\n                                   '
                      '(Angstrom^3)         (eV)             (eV)          (eV/Angstrom)          (mu_B)\n', False,
            True, True)

    original_stdout = sys.stdout
    buffer = io.StringIO()
    sys.stdout = buffer

    conv_bool_f = True
    filter_f = partial(FrechetCellFilter, mask=list(dof_array))
    relaxer = StructOptimizer(model=model_chgnet, optimizer_class=ibrion)
    result = relaxer.relax(structure_f, fmax=ediffg, steps=nsw, relax_cell=isif, ase_filter=filter_f, loginterval=ofi)

    result_coord = result["final_structure"].frac_coords
    if output_contcar:
        atom_name_f, atom_num_f = wr_atom_list(result["final_structure"].atomic_numbers)
        if "selective_dynamics" in structure_f.site_properties:
            coord_relax_f = np.array(structure_f.site_properties["selective_dynamics"])
        else:
            coord_relax_f = np.zeros(result_coord.shape) + 1

        wr_contcar(path_job, "CONTCAR", Title, result["final_structure"].lattice.matrix,
                   result_coord, atom_name_f, atom_num_f, coord_relax_f, None)
    traj_f = result["trajectory"]
    if len(traj_f) > nsw + 1:
        conv_bool_f = False
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
                wr_xdatcar_head(path_job, "Trajectory_VASP", lc_mat_f, Title, Atom_type, Atom_num)
            wr_xdatcar_coord(path_job, "Trajectory_VASP", nif * OFI_OPT, coord_fraction_f)

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
        ax1.plot(step_f, (energy_f - energy_f[0]) * traj_f.atom_positions[0].shape[0],
                 label="Total energy difference (eV)", color="r")
        ax1.set_xlabel("Steps", fontsize=20, color="k", fontweight="bold")
        ax1.set_ylabel("Total energy difference", fontsize=20, color="r", fontweight="bold")
        plt.legend(fontsize=16, loc='upper left')
        ax2 = ax1.twinx()
        ax2.plot(step_f, force_f, label="Force_max (eV/Angstrom)", color="b")
        ax2.set_ylabel("Force_max (eV/Angstrom)", fontsize=20, color="b", fontweight="bold")
        ax2.set_yscale("log")
        plt.title("Energy & Force vs. Steps", fontsize=24, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=16, loc='upper right')
        plt.savefig(os.path.join(path_job, "_CHGNet_Optimization_Energy_Force.png"), dpi=200)
        plt.close()

    result_structure_f = result["final_structure"]
    if "selective_dynamics" in structure_f.site_properties:
        result_structure_f.add_site_property("selective_dynamics",
                                             structure_f.site_properties["selective_dynamics"])

    return (result["trajectory"].energies[-1], result_structure_f, conv_bool_f,
            traj_f.stresses[len(traj_f.stresses) - 1] * 160.21766208)


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


def cal_phonon(path_job, model_chgnet, structure_origin_f, temp_sta, ts_cal,
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
        gen(Path_log, "\n", True, False, False)

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
    if ts_cal & (vib_freq_f[-1] < 0):
        vib_freq_f_save = vib_freq_f_save[: -1]
    vib_energy = h_cons * vib_freq_f_save / q_cons  # in eV
    vib_zpe_f = np.sum(vib_energy) / 2  # in eV

    vib_internal_energy_f = np.sum(vib_energy / (np.exp(vib_energy * q_cons / (kB_cons * temp_sta)) - 1))

    vib_temp_ratio_f = h_cons * vib_freq_f_save / (kB_cons * temp_sta)
    vib_entropy_f = kB_cons / q_cons * np.sum(vib_temp_ratio_f / (np.exp(vib_temp_ratio_f) - 1) -
                                              np.log(1 - np.exp(-vib_temp_ratio_f)))  # in eV / K

    if output_thermal_results:
        path_log = os.path.join(path_job, "CHGNet_results.log")
        if not os.path.exists(path_log):
            path_log = os.path.join(path_job, "CHGNet_phonon_results.log")
            with open(path_log, 'w') as nf_rf:
                nf_rf.write("# " + st_str + "\n\n")

        gen(path_log, "", True, False, True)
        gen(path_log, "The phonon calculation is complete. A total of " +
            format(dof_f - num_image_freq_f, '.0f') + " real frequencies and " +
            format(num_image_freq_f, '.0f') + " imaginary frequencies were found.",
            True, True, True)
        if ts_cal:
            gen(path_log, "The program corrects the " + format(num_real_low_freq_f + num_image_freq_f, '.0f') +
                " frequencies with wave numbers below " + format(CUT_WN, '.1f') +
                " cm-1 to obtain the thermodynamic quantities at " + format(temp_sta, '.1f') +
                " K: (neglected the maximum negative frequency)\n", True, True, True)
        else:
            gen(path_log, "The program corrects the " + format(num_real_low_freq_f + num_image_freq_f, '.0f') +
                " frequencies with wave numbers below " + format(CUT_WN, '.1f') +
                " cm-1 to obtain the thermodynamic quantities at " + format(temp_sta, '.1f') +
                " K:\n", True, True, True)

        energy_corr += vib_zpe_f
        gen(path_log, "Zero-point energy ZPE     " +
            num_printer(vib_zpe_f * q_cons * NA_cons / kcal_to_J_cons, 15, 3, 0) + " kcal/mol, " +
            num_printer(vib_zpe_f, 10, 4, 0) + " eV" +
            num_printer(vib_zpe_f / num_atom, 14, 6, 0) +
            " eV/atoms\n                         Corrected energy, E" +
            num_printer(energy_corr, 43, 3, 0) + " eV" +
            num_printer(energy_corr / num_atom, 14, 6, 0) + " eV/atoms",
            True, True, True)
        energy_corr += vib_internal_energy_f
        gen(path_log, "Thermal correction to U(T)" +
            num_printer((vib_zpe_f + vib_internal_energy_f) *
                        q_cons * NA_cons / kcal_to_J_cons, 15, 3, 0) + " kcal/mol, " +
            num_printer(vib_zpe_f + vib_internal_energy_f, 10, 4, 0) + " eV" +
            num_printer((vib_zpe_f + vib_internal_energy_f) / num_atom, 14, 6, 0) +
            " eV/atoms\n                         Corrected energy, U" +
            num_printer(energy_corr, 43, 3, 0) + " eV" +
            num_printer(energy_corr / num_atom, 14, 6, 0) + " eV/atoms",
            True, True, True)
        energy_corr -= vib_entropy_f * temp_sta
        gen(path_log, "Thermal correction to G(T)" +
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
        gen(path_log, "Entropy S                 " +
            num_printer(vib_entropy_f * q_cons * NA_cons / 1000, 15, 3, 0) + " kJ/mol  , " +
            num_printer(vib_entropy_f, 10, 4, 0) + " eV/K" +
            num_printer(vib_entropy_f / num_atom, 12, 6, 0) + " eV/atoms/K",
            True, True, True)
        gen(path_log, "Entropy contribution T*S  " +
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


def cal_vec_rotate(normal_vec_f, da_f, vec_rot_origin_f, index_f1, index_f2):
    vec_rot_f = vec_rot_origin_f.copy()
    sin_angle_rf, cos_angle_rf = np.sin(da_f), np.cos(da_f)
    # Rodrigues' rotation formula
    replace_mat_f = (
        np.array([vec_rot_f[index_f1, :] * cos_angle_rf +
                  np.cross(normal_vec_f, vec_rot_f[index_f1, :]) * sin_angle_rf +
                  normal_vec_f * np.dot(normal_vec_f, vec_rot_f[index_f1, :]) * (1 - cos_angle_rf),
                  vec_rot_f[index_f2, :] * cos_angle_rf -
                  np.cross(normal_vec_f, vec_rot_f[index_f2, :]) * sin_angle_rf +
                  normal_vec_f * np.dot(normal_vec_f, vec_rot_f[index_f2, :]) * (1 - cos_angle_rf)]))

    angle_r_f = np.arccos(np.dot(replace_mat_f[0, :], replace_mat_f[1, :]) /
                          np.linalg.norm(replace_mat_f[0, :]) / np.linalg.norm(replace_mat_f[1, :]))
    vec_rot_f[index_f1, :], vec_rot_f[index_f2, :] = replace_mat_f[0, :], replace_mat_f[1, :]

    return vec_rot_f, angle_r_f


def cal_stress(model_chgnet, path_res, struc_origin_f, lc_mat_f_p2, lc_mat_f_p1, lc_mat_f_n1, lc_mat_f_n2):
    struc_f = struc_origin_f.copy()
    prop_f = struc_f.site_properties
    new_struc_f_p2 = Structure(Lattice(lc_mat_f_p2), struc_f.species, struc_f.frac_coords)
    new_struc_f_p1 = Structure(Lattice(lc_mat_f_p1), struc_f.species, struc_f.frac_coords)
    new_struc_f_n1 = Structure(Lattice(lc_mat_f_n1), struc_f.species, struc_f.frac_coords)
    new_struc_f_n2 = Structure(Lattice(lc_mat_f_n2), struc_f.species, struc_f.frac_coords)
    if "selective_dynamics" in prop_f.keys():
        new_struc_f_p2.add_site_property("selective_dynamics", prop_f["selective_dynamics"])
        new_struc_f_p1.add_site_property("selective_dynamics", prop_f["selective_dynamics"])
        new_struc_f_n1.add_site_property("selective_dynamics", prop_f["selective_dynamics"])
        new_struc_f_n2.add_site_property("selective_dynamics", prop_f["selective_dynamics"])
        coord_relax_f = np.array(struc_origin_f.site_properties["selective_dynamics"])
    else:
        coord_relax_f = np.zeros(struc_origin_f.frac_coords.shape) + 1

    path_sub = [path_res + "_positive_2", path_res + "_positive_1",
                path_res + "_negative_1", path_res + "_negative_2"]
    lc_mat_list_f = [lc_mat_f_p2, lc_mat_f_p1, lc_mat_f_n1, lc_mat_f_n2]
    new_struc_f = [new_struc_f_p2, new_struc_f_p1, new_struc_f_n1, new_struc_f_n2]
    stress_f = np.zeros((4, 6))
    for nif_job in range(4):
        if not os.path.exists(path_sub[nif_job]):
            os.mkdir(path_sub[nif_job])

        wr_contcar(path_sub[nif_job], "POSCAR", Title, lc_mat_list_f[nif_job], struc_f.frac_coords,
                   Atom_type, Atom_num, coord_relax_f, np.zeros((struc_f.frac_coords.shape[0], 3)))
        _, _, conv_bool_f, stress_f[nif_job, :] = (
            cal_opt(model_chgnet, new_struc_f[nif_job], path_sub[nif_job], IBRION, EDIFFG_M, NSW_MP, False, OFI_MP,
                    np.array([DOF_A, DOF_B, DOF_C, DOF_BC, DOF_AC, DOF_AB]),
                    True, True, True, True, True))  # in GPa

        gen(Path_log, "Calculation progress: " + os.path.basename(path_sub[nif_job]),
            True, True, False)
        if conv_bool_f:
            gen(Path_log, "  => Reach convergence", True, False, True)
        else:
            gen(Path_log, "  => Not yet converged", True, False, True)

    gen(Path_log, "", True, False, True)

    return stress_f


def cal_elas_tensor(model_chgnet, struc_f, ratio_f, path_tmp):
    gen(Path_log, 'Reading completed, calculation method is "Mechanical Property Calculation".\n',
        True, True, True)
    elas_mat = np.zeros((6, 6))  # elasticity tensor
    lc_mat_origin = struc_f.lattice.matrix
    # for strain of a, b, c
    path_res = [os.path.join(path_tmp, "Normal_strain__A"), os.path.join(path_tmp, "Normal_strain__B"),
                os.path.join(path_tmp, "Normal_strain__C"), os.path.join(path_tmp, "Shear_strain__BC"),
                os.path.join(path_tmp, "Shear_strain__AC"), os.path.join(path_tmp, "Shear_strain__AB")]
    for nif_vec in range(3):
        lc_mat_p2, lc_mat_p, lc_mat_n, lc_mat_n2 = (
            lc_mat_origin.copy(), lc_mat_origin.copy(), lc_mat_origin.copy(), lc_mat_origin.copy())
        lc_mat_p2[nif_vec, :] *= 1 + 2 * ratio_f
        lc_mat_p[nif_vec, :] *= 1 + ratio_f
        lc_mat_n[nif_vec, :] *= 1 - ratio_f
        lc_mat_n2[nif_vec, :] *= 1 - 2 * ratio_f

        stress_mat_f = cal_stress(model_chgnet, path_res[nif_vec], struc_f, lc_mat_p2, lc_mat_p, lc_mat_n, lc_mat_n2)
        elas_mat[:, nif_vec] = -(stress_mat_f[0, :] - 8 * stress_mat_f[1, :] +
                                 8 * stress_mat_f[2, :] - stress_mat_f[3, :]) / (12 * ratio_f)

    for nif_shear in range(3):
        if nif_shear == 0:
            index_f1, index_f2 = 1, 2
        elif nif_shear == 1:
            index_f1, index_f2 = 0, 2
        else:
            index_f1, index_f2 = 0, 1

        # for strain of gamma_BC
        vec_1, vec_2 = lc_mat_origin[index_f1, :], lc_mat_origin[index_f2, :]
        vec_1_len, vec_2_len = np.linalg.norm(vec_1), np.linalg.norm(vec_2)
        rotate_axis = np.cross(vec_1, vec_2)
        rotate_axis /= np.linalg.norm(rotate_axis)
        angle_eq = np.arccos(np.dot(vec_1, vec_2) / (vec_1_len * vec_2_len))

        lc_mat_a2, angle_r_a2 = cal_vec_rotate(rotate_axis, angle_eq * RATIO_MP,
                                               lc_mat_origin, index_f1, index_f2)
        lc_mat_a, angle_r_a = cal_vec_rotate(rotate_axis, angle_eq * RATIO_MP / 2,
                                             lc_mat_origin, index_f1, index_f2)
        lc_mat_b, angle_r_b = cal_vec_rotate(rotate_axis, -angle_eq * RATIO_MP / 2,
                                             lc_mat_origin, index_f1, index_f2)
        lc_mat_b2, angle_r_b2 = cal_vec_rotate(rotate_axis, -angle_eq * RATIO_MP,
                                               lc_mat_origin, index_f1, index_f2)

        gamma_f = np.cos(angle_r_a) / np.sin(angle_r_a) - np.cos(angle_r_b) / np.sin(angle_r_b)
        stress_mat_f = cal_stress(model_chgnet, path_res[3 + nif_shear],
                                  struc_f, lc_mat_a2, lc_mat_a, lc_mat_b, lc_mat_b2)
        elas_mat[:, 3 + nif_shear] = -(stress_mat_f[0, :] - 8 * stress_mat_f[1, :] +
                                       8 * stress_mat_f[2, :] - stress_mat_f[3, :]) / (6 * gamma_f * 2)

    t_mat_f = np.zeros((6, 6))
    lc_f = lc_mat_origin / np.linalg.norm(lc_mat_origin, axis=1)[:, np.newaxis]
    for nif_t in range(3):
        t_mat_f[nif_t, :] = (
            np.array([lc_f[0, nif_t] ** 2, lc_f[1, nif_t] ** 2, lc_f[2, nif_t] ** 2,
                      2 * lc_f[1, nif_t] * lc_f[2, nif_t],
                      2 * lc_f[0, nif_t] * lc_f[2, nif_t],
                      2 * lc_f[0, nif_t] * lc_f[1, nif_t]]))
    for nif_t in range(3):
        t_mat_f[3:, nif_t] = (
                2 * np.array([lc_f[nif_t, 1] * lc_f[nif_t, 2],
                              lc_f[nif_t, 0] * lc_f[nif_t, 2],
                              lc_f[nif_t, 0] * lc_f[nif_t, 1]]))
    t_mat_f[3:, 3:] = (
            2 * np.array([
        [lc_f[1, 1] * lc_f[2, 2] + lc_f[2, 1] * lc_f[1, 2],
         lc_f[0, 1] * lc_f[2, 2] + lc_f[2, 1] * lc_f[0, 2],
         lc_f[0, 1] * lc_f[1, 2] + lc_f[1, 1] * lc_f[0, 2]],
        [lc_f[1, 0] * lc_f[2, 2] + lc_f[2, 0] * lc_f[1, 2],
         lc_f[0, 0] * lc_f[2, 2] + lc_f[2, 0] * lc_f[0, 2],
         lc_f[0, 0] * lc_f[1, 2] + lc_f[1, 0] * lc_f[0, 2]],
        [lc_f[1, 0] * lc_f[2, 1] + lc_f[2, 0] * lc_f[1, 1],
         lc_f[0, 0] * lc_f[2, 1] + lc_f[2, 0] * lc_f[0, 1],
         lc_f[0, 0] * lc_f[1, 1] + lc_f[1, 0] * lc_f[0, 1]]]))

    elas_mat_abc = (elas_mat + elas_mat.T) / 2
    elas_mat_xyz = np.dot(np.dot(t_mat_f, elas_mat_abc), t_mat_f.T)

    return ElasticTensor.from_voigt(elas_mat_abc), ElasticTensor.from_voigt(elas_mat_xyz)


def wr_mech_prop(path_log, elas_mat_abc, elas_mat_xyz):
    gen(path_log, "", True, True, True)
    gen(path_log, "The calculation is completed and "
                  "the mechanical properties are sorted out...\n", True, True, True)
    gen(path_log, "***********************************"
                  "*******************************", True, True, True)
    gen(path_log, "The Elastic Tensor (GPa): ", True, True, True)
    elas_voigt = elas_mat_abc.voigt
    gen(path_log, "                                         AA             BB             CC"
                  "             BC             AC             AB    ", True, False, True)
    for ni_row in range(6):
        if ni_row == 0:
            gen(path_log, "  AA  ", True, True, False)
        elif ni_row == 1:
            gen(path_log, "  BB  ", True, True, False)
        elif ni_row == 2:
            gen(path_log, "  CC  ", True, True, False)
        elif ni_row == 3:
            gen(path_log, "  BC  ", True, True, False)
        elif ni_row == 4:
            gen(path_log, "  AC  ", True, True, False)
        elif ni_row == 5:
            gen(path_log, "  AB  ", True, True, False)

        for ni_col in range(6):
            gen(path_log, num_printer(elas_voigt[ni_row, ni_col], 15, 6, 0),
                True, False, False)
        gen(path_log, "", True, False, True)

    elas_voigt = elas_mat_xyz.voigt
    gen(path_log, "                                         XX             YY             ZZ"
                  "             YZ             XZ             XY    ", True, False, True)
    for ni_row in range(6):
        if ni_row == 0:
            gen(path_log, "  XX  ", True, True, False)
        elif ni_row == 1:
            gen(path_log, "  YY  ", True, True, False)
        elif ni_row == 2:
            gen(path_log, "  ZZ  ", True, True, False)
        elif ni_row == 3:
            gen(path_log, "  YZ  ", True, True, False)
        elif ni_row == 4:
            gen(path_log, "  XZ  ", True, True, False)
        elif ni_row == 5:
            gen(path_log, "  XY  ", True, True, False)

        for ni_col in range(6):
            gen(path_log, num_printer(elas_voigt[ni_row, ni_col], 15, 6, 0),
                True, False, False)
        gen(path_log, "", True, False, True)

    gen(path_log, "", True, False, True)
    gen(path_log, "Shear modulus (G)  (Average):  " +
        num_printer(Elas_mat_xyz.g_voigt, 12, 6, 0) + " (Voigt)" +
        num_printer(Elas_mat_xyz.g_reuss, 12, 6, 0) + " (Reuss)" +
        num_printer(Elas_mat_xyz.g_vrh, 12, 6, 0) + " (Voigt-Reuss-Hill)  GPa",
        True, True, True)
    gen(path_log, "Bulk modulus (K)   (Average):  " +
        num_printer(Elas_mat_xyz.k_voigt, 12, 6, 0) + " (Voigt)" +
        num_printer(Elas_mat_xyz.k_reuss, 12, 6, 0) + " (Reuss)" +
        num_printer(Elas_mat_xyz.k_vrh, 12, 6, 0) + " (Voigt-Reuss-Hill)  GPa",
        True, True, True)

    gen(path_log, "", True, False, True)
    gen(path_log, "Young's modulus (E) (Average): " +
        num_printer(Elas_mat_xyz.y_mod * 1E-9, 12, 6, 0) + "  GPa",
        True, True, True)
    gen(path_log, "Young's modulus (E):           " +
        num_printer(Elas_mat_abc.directional_elastic_mod([1, 0, 0]), 12, 6, 0) + " (A)  " +
        num_printer(Elas_mat_abc.directional_elastic_mod([0, 1, 0]), 12, 6, 0) + " (B)  " +
        num_printer(Elas_mat_abc.directional_elastic_mod([0, 0, 1]), 12, 6, 0) + " (C)  GPa",
        True, True, True)
    gen(path_log, "Young's modulus (E):           " +
        num_printer(Elas_mat_xyz.directional_elastic_mod([1, 0, 0]), 12, 6, 0) + " (x)  " +
        num_printer(Elas_mat_xyz.directional_elastic_mod([0, 1, 0]), 12, 6, 0) + " (y)  " +
        num_printer(Elas_mat_xyz.directional_elastic_mod([0, 0, 1]), 12, 6, 0) + " (z)  GPa",
        True, True, True)

    gen(path_log, "", True, False, True)
    gen(path_log, "Poisson's ratio (Gamma) (Average): " +
        num_printer(Elas_mat_xyz.homogeneous_poisson, 12, 6, 0),
        True, True, True)
    gen(path_log, "Poisson's ratio (Gamma):           " +
        num_printer(Elas_mat_abc.directional_poisson_ratio([0, 1, 0],
                                                           [0, 0, 1]), 12, 6, 0) + " (BC)  " +
        num_printer(Elas_mat_abc.directional_poisson_ratio([1, 0, 0],
                                                           [0, 0, 1]), 12, 6, 0) + " (AC)  " +
        num_printer(Elas_mat_abc.directional_poisson_ratio([1, 0, 0],
                                                           [0, 1, 0]), 12, 6, 0) + " (AB)  ",
        True, True, True)
    gen(path_log, "Poisson's ratio (Gamma):           " +
        num_printer(Elas_mat_xyz.directional_poisson_ratio([0, 1, 0],
                                                           [0, 0, 1]), 12, 6, 0) + " (YZ)  " +
        num_printer(Elas_mat_xyz.directional_poisson_ratio([1, 0, 0],
                                                           [0, 0, 1]), 12, 6, 0) + " (XZ)  " +
        num_printer(Elas_mat_xyz.directional_poisson_ratio([1, 0, 0],
                                                           [0, 1, 0]), 12, 6, 0) + " (XY)  ",
        True, True, True)

    gen(path_log, "", True, False, True)
    gen(path_log, "Transverse sound velocity (Average):   " +
        num_printer(Elas_mat_xyz.trans_v(STRUCture), 10, 4, 0) + " m/s",
        True, True, True)
    gen(path_log, "Longitudinal sound velocity (Average): " +
        num_printer(Elas_mat_xyz.long_v(STRUCture), 10, 4, 0) + " m/s",
        True, True, True)
    gen(path_log, "Snyder's sound velocity (Average):     " +
        num_printer(Elas_mat_xyz.snyder_total(STRUCture), 10, 4, 0) + " (total)  " +
        num_printer(Elas_mat_xyz.snyder_ac(STRUCture), 10, 4, 0) + " (acoustic)  " +
        num_printer(Elas_mat_xyz.snyder_opt(STRUCture), 10, 4, 0) + " (optical)   m/s",
        True, True, True)

    gen(path_log, "", True, False, True)
    gen(path_log, "Debye temperature:    " +
        num_printer(Elas_mat_xyz.debye_temperature(STRUCture), 10, 4, 0) + " K",
        True, True, True)
    gen(path_log, "Thermal conductivity: " +
        num_printer(Elas_mat_xyz.clarke_thermalcond(STRUCture), 10, 4, 0) + " (Clarke's)  " +
        num_printer(Elas_mat_xyz.cahill_thermalcond(STRUCture), 10, 4, 0) + " (Cahill's)  " +
        num_printer(Elas_mat_xyz.agne_diffusive_thermalcond(STRUCture), 10, 4, 0) + " (Agne's)   W/(m*K)",
        True, True, True)

    gen(path_log, "", True, False, True)
    gen(path_log, "Universal Elastic Anisotropy Index (A_U): " +
        num_printer(Elas_mat_xyz.universal_anisotropy, 10, 3, 0), True, True, True)


def select_index(atom_doped_mat_origin_f, faseq_array_index_f):
    atom_doped_mat_f = atom_doped_mat_origin_f.copy()
    index_s1 = np.random.randint(faseq_array_index_f.shape[0])  # choose 1st index
    filtered_index_s1 = np.delete(np.arange(0, faseq_array_index_f.shape[0], 1),
                                  np.where(atom_doped_mat_f == atom_doped_mat_f[index_s1])[0])
    # filtered_index_s1: list the index without 1st kind of element
    index_s2 = np.random.choice(filtered_index_s1)
    element_s1_f = atom_doped_mat_f[index_s2]
    atom_doped_mat_f[index_s2] = atom_doped_mat_f[index_s1]
    atom_doped_mat_f[index_s1] = element_s1_f

    return atom_doped_mat_f


def replace_atom(struc_ori, atom_doped_mat, faseq_array_index):
    struc_f = struc_ori.copy()
    for nif, nif_index in enumerate(faseq_array_index):
        if atom_doped_mat[nif] == "Va":
            struc_f.replace(idx=int(nif_index), species=DummySpecies("X"))
        else:
            struc_f.replace(idx=int(nif_index), species=atom_doped_mat[nif])

    for nif_key, nif_prop in STRUCture_site_prop.items():
        struc_f.add_site_property(nif_key, nif_prop)

    return struc_f


def replace_atom_beginning(struc_ori, faseq_array_index, name_occ_f, num_occ_f):
    atom_doped_label_mat = np.zeros((len(name_occ_f), 2), dtype=object)
    for nif in range(atom_doped_label_mat.shape[0]):
        atom_doped_label_mat[nif, 0] = name_occ_f[nif]
        atom_doped_label_mat[nif, 1] = int(num_occ_f[nif])
    atom_doped_mat = np.zeros(int(np.sum(atom_doped_label_mat[:, 1])), dtype=object)
    for nif in range(len(name_occ_f)):
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
    atom_doped_mat_save_f = {}
    structure_save_f, structure_save_opted_f, structure_screen_f = {}, {}, {}
    st_time_mc = time.time()
    num_occ_f = np.sum(Atom_num)
    for nif_key, nif_str_array in FASEQ_array_index_dict.items():
        atom_doped_mat_save_f[nif_key] = np.zeros((int(NSW_MC_G), int(NSW_MC_S) + 1,
                                                   nif_str_array.shape[0]), dtype=object)
        name_occ_fi, num_occ_fi = NAME_OCC_dict[nif_key], NUM_OCC_dict[nif_key]
        structure_f, atom_doped_mat_f = replace_atom_beginning(structure_f, nif_str_array,
                                                               name_occ_fi, num_occ_fi)  # initialization
        atom_doped_mat_save_f[nif_key][:, 0, :] = atom_doped_mat_f
        num_occ_f -= np.sum(atom_doped_mat_f == "Va")
    num_occ_f = int(num_occ_f)

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
               atom_name_mc_f, atom_num_mc_f, relax_atom_f, None)

    structure_clear_f = rm_dummy_atom(structure_f)
    energy_initial_f, structure_clear_opted_0_f, _, _ = (
        cal_opt(Model_CHGNET, structure_clear_f, path_res_sub, IBRION, EDIFFG1, NSW_OPT1, ISIF1, OFI_OPT1,
                np.array([DOF_A, DOF_B, DOF_C, DOF_BC, DOF_AC, DOF_AB]),
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
        "\n\n                            Group-Iteration   Sampling type   Energy (eV/atoms)     dE (eV)    "
        "Acceptance rate     State\n", True, True, True)
    energy_save_mat_f[:, 0] = energy_initial_f / num_occ_f
    energy_save_f[:, 0] = energy_initial_f / num_occ_f

    hash_mat_f = np.zeros((NSW_MC_G, NSW_MC_S + 1), dtype=object)
    hash_mat_f[:, 0] = get_structure_hash(structure_clear_f)

    # 1st screening
    for nif_main in range(int(NSW_MC_G)):  # main loop
        structure_save_f[nif_main], structure_save_opted_f[nif_main] = {}, {}
        path_res_main = os.path.join(Path_tmp, "tmp_" + str(nif_main + 1))
        if not os.path.exists(path_res_main):
            os.mkdir(path_res_main)

        structure_f = structure_origin_f.copy()
        for nif_key, nif_str_array in FASEQ_array_index_dict.items():
            name_occ_fi, num_occ_fi = NAME_OCC_dict[nif_key], NUM_OCC_dict[nif_key]
            structure_f, atom_doped_mat_f = replace_atom_beginning(structure_f, nif_str_array,
                                                                   name_occ_fi, num_occ_fi)  # initialization
        structure_save_f[nif_main][0] = structure_f
        structure_save_opted_f[nif_main][0] = structure_clear_opted_0_f
        structure_accept_f = structure_f.copy()

        # choose the MC group
        mc_sampling_f = np.zeros(len(FASEQ_array_index_dict))
        mc_sampling_key_f = np.array([nif for nif in FASEQ_array_index_dict.keys()])
        mc_sampling_list = np.zeros(int(NSW_MC_S), dtype=object)
        min_r_mc_num_f = np.floor(MIN_R_MC * NSW_MC_S)
        for nif in range(int(NSW_MC_S)):
            mc_sampling_list[nif] = np.random.choice(mc_sampling_key_f[mc_sampling_f < min_r_mc_num_f])
            mc_sampling_f[mc_sampling_list[nif] == mc_sampling_key_f] += 1
            if np.sum(mc_sampling_f < min_r_mc_num_f) == 0:
                break

        num_left_group_f = int(NSW_MC_S - np.sum(mc_sampling_f))
        if num_left_group_f > 0:
            mc_sampling_list[int(np.sum(mc_sampling_f)):] = (
                mc_sampling_key_f)[np.random.randint(Num_MC_group, size=num_left_group_f)]

        # initialization
        structure_find_index_f, structure_find_num_f, structure_find_num_save_f = 0, 1, np.array([0])
        convg_num_f = int(max([NSW_MC_S / 2, 5]))
        for nif_sub in range(1, int(NSW_MC_S) + 1):  # minor loop
            path_res_sub = os.path.join(path_res_main, "tmp_" + str(nif_main + 1) + "_" + str(nif_sub))
            if not os.path.exists(path_res_sub):
                os.mkdir(path_res_sub)

            # get the "POSCAR"
            for nif_key, nif_str_array in atom_doped_mat_save_f.items():
                atom_doped_mat_save_f[nif_key][nif_main, nif_sub, :] = nif_str_array[nif_main, nif_sub - 1, :]
            group_key_f = mc_sampling_list[nif_sub - 1]
            faseq_array_index_f = FASEQ_array_index_dict[group_key_f]
            atom_doped_mat_f = (
                select_index(atom_doped_mat_save_f[group_key_f][nif_main, nif_sub, :], faseq_array_index_f))
            structure_f = replace_atom(structure_accept_f, atom_doped_mat_f, faseq_array_index_f)
            structure_save_f[nif_main][nif_sub] = structure_f
            structure_clear_f = rm_dummy_atom(structure_f)
            hash_v = get_structure_hash(structure_clear_f)

            if np.sum(hash_v == hash_mat_f) != 0:  # has been calculated
                index_hash_f = np.argwhere(hash_v == hash_mat_f)[0, :]
                energy_save_mat_f[nif_main, nif_sub] = energy_save_mat_f[index_hash_f[0], index_hash_f[1]]
                structure_save_opted_f[nif_main][nif_sub] = structure_save_opted_f[index_hash_f[0]][index_hash_f[1]]
                pre_cal_state_f = " (Precomputation)"
            else:
                # calculate the energy
                atom_name_mc_f, atom_num_mc_f = wr_atom_list(structure_f.atomic_numbers)
                relax_atom_f = np.zeros(structure_f.frac_coords.shape) + 1
                if "selective_dynamics" in structure_f.site_properties:
                    relax_atom_f = np.array(structure_f.site_properties["selective_dynamics"])
                wr_contcar(path_res_sub, "POSCAR", Title, structure_f.lattice.matrix, structure_f.frac_coords,
                           atom_name_mc_f, atom_num_mc_f, relax_atom_f, None)
                energy_system_f, structure_clear_opted_f, _, _ = (
                    cal_opt(Model_CHGNET, structure_clear_f, path_res_sub, IBRION, EDIFFG1, NSW_OPT1, ISIF1, OFI_OPT1,
                            np.array([DOF_A, DOF_B, DOF_C, DOF_BC, DOF_AC, DOF_AB]), True, True,
                            False, False, False))
                energy_save_mat_f[nif_main, nif_sub] = energy_system_f / num_occ_f
                structure_save_opted_f[nif_main][nif_sub] = structure_clear_opted_f
                pre_cal_state_f = "                 "

            hash_mat_f[nif_main, nif_sub] = hash_v

            delta_e_f = energy_save_mat_f[nif_main, nif_sub] - energy_save_mat_f[nif_main, structure_find_index_f]
            delta_e_joule = delta_e_f * num_occ_f * q_cons
            temp_mc_f = TEMP_MCI + (TEMP_MCF - TEMP_MCI) * nif_sub / NSW_MC_S
            acceptance_rate_f = min([1, np.exp(-delta_e_joule / kB_cons / temp_mc_f)])
            if acceptance_rate_f >= np.random.rand():
                state_mc_f = "Acceptance"
                structure_find_index_f = nif_sub
                energy_save_f[nif_main, structure_find_num_f] = energy_save_mat_f[nif_main, nif_sub]
                structure_find_num_f += 1
                structure_find_num_save_f = (
                    np.concatenate((structure_find_num_save_f, np.array([nif_sub])), axis=0))
                structure_accept_f = structure_f.copy()
                atom_doped_mat_save_f[group_key_f][nif_main, nif_sub, :] = atom_doped_mat_f

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

                path_copy = os.path.join(path_res_sub, "POSCAR")
                if os.path.exists(path_copy):
                    shutil.copy(path_copy, os.path.join(Path_job, "POSCAR_current"))
                    
                path_copy = os.path.join(path_res_sub, "CONTCAR")
                if os.path.exists(path_copy):
                    shutil.copy(path_copy, os.path.join(Path_job, "CONTCAR_current"))

            else:
                state_mc_f = "*Rejection"

            gen(path_log,
                num_printer(nif_main + 1, 8, 0, 0) + " - " +
                num_printer(nif_sub, 8, 0, 0) + MC_Key_dict[group_key_f] +
                num_printer(energy_save_mat_f[nif_main, nif_sub], 20, 6, 0) +
                num_printer(delta_e_f * num_occ_f, 16, 3, 0) +
                num_printer(acceptance_rate_f * 100, 14, 2, 0) + "%        " +
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

    path_sampling = os.path.join(Path_job, "CHGNet_sampling")
    if not os.path.exists(path_sampling):
        os.mkdir(path_sampling)

    energy_diff_mat_f = (energy_save_mat_f - np.min(energy_save_mat_f)) * num_occ_f  # in eV
    data_argsort_f = np.argsort(np.argsort(energy_diff_mat_f.flatten())).reshape(energy_diff_mat_f.shape)
    num_sampling_all_f = int(np.sum(energy_diff_mat_f <= E_SACONF))
    num_conf_f = min([NUM_CONF, num_sampling_all_f])
    sampling_dist_f = np.logspace(0, np.log(num_sampling_all_f), num_conf_f, endpoint=True, base=np.exp(1)) - 1
    for nif_samp in range(1, sampling_dist_f.shape[0]):
        sampling_dist_f[nif_samp] = max([sampling_dist_f[nif_samp - 1] + 1, np.floor(sampling_dist_f[nif_samp])])

    energy_grep_save_f = np.zeros(sampling_dist_f.shape[0])
    for nif_seq, nif_samp in enumerate(sampling_dist_f):
        hf1, hf2 = np.where(data_argsort_f == nif_samp)
        hf1, hf2 = hf1[0], hf2[0]
        struc_f = structure_save_opted_f[hf1][hf2]
        file_name_poscar_f = ("POSCAR_" + format(nif_seq + 1, '.0f') + "_" +
                              format(energy_diff_mat_f[hf1, hf2] * 1000, '.0f') + "meV")
        energy_grep_save_f[nif_seq] = energy_diff_mat_f[hf1, hf2]
        atom_name_mc_f, atom_num_mc_f = wr_atom_list(struc_f.atomic_numbers)
        relax_atom_f = np.zeros(struc_f.frac_coords.shape) + 1
        if "selective_dynamics" in struc_f.site_properties:
            relax_atom_f = np.array(struc_f.site_properties["selective_dynamics"])
        wr_contcar(path_sampling, file_name_poscar_f, Title, struc_f.lattice.matrix, struc_f.frac_coords,
                   atom_name_mc_f, atom_num_mc_f, relax_atom_f, None)

    shutil.make_archive(path_sampling, 'gztar', path_sampling)
    if os.path.exists(path_sampling):
        shutil.rmtree(path_sampling)

    bin_f = np.max(energy_diff_mat_f) / min([20, np.prod(energy_diff_mat_f.shape)])
    bin_f = np.ceil(bin_f * 10) / 10
    bin_array_f = np.arange(0, np.max(energy_diff_mat_f) + bin_f, bin_f)
    hist_data_f, _ = np.histogram(energy_diff_mat_f.flatten(), bins=bin_array_f)
    bin_array_f = (bin_array_f[1:] + bin_array_f[: -1]) / 2 * 1000
    plt.figure(figsize=(16, 9))
    cmap_f = plt.get_cmap('bwr')
    normalize_f = plt.Normalize(bin_array_f[0], bin_array_f[-1])
    plt.bar(bin_array_f, hist_data_f / 100, width=bin_f * 1000 / 1.5, edgecolor='black',
            color=cmap_f(normalize_f(bin_array_f)))
    plt.xlabel("Energy Difference (meV)")
    plt.ylabel("Density of state")
    plt.grid(True, linestyle="--")
    plt.savefig(os.path.join(Path_job, "Sampling distribution.png"), dpi=300)
    plt.close()

    energy_grep_save_f *= 1000
    plt.figure(figsize=(16, 9))
    plt.plot(np.arange(0, energy_grep_save_f.shape[0], 1), energy_grep_save_f, color='k', linestyle='--')
    plt.scatter(np.arange(0, energy_grep_save_f.shape[0], 1), energy_grep_save_f, color='r')
    plt.xlabel("Output configuration number")
    plt.ylabel("Energy difference (meV)")
    plt.grid(True, linestyle="--")
    plt.savefig(os.path.join(Path_job, "Energy distribution.png"), dpi=300)
    plt.close()

    return num_occ_f, energy_save_mat_f, structure_screen_f, atom_doped_mat_save_f


def cal_mc_opt(path_log, path_tmp):
    gen(path_log, "", True, True, True)
    gen(path_log, "Start secondary screening\n"
                  "                            Group/Iteration"
                  "           Energy (eV/atoms)           Structure\n", True, True, True)

    energy_screen_mat_f = np.zeros((int(NSW_MC_G), int(SCQU1)))
    index_screen_mat_f = np.zeros((int(NSW_MC_G), int(SCQU1)))
    num_occ_f = np.sum(Atom_num)
    for nif_key, nif_str_array in FASEQ_array_index_dict.items():
        name_occ_fi, num_occ_fi = NAME_OCC_dict[nif_key], NUM_OCC_dict[nif_key]
        if "Va" in name_occ_fi:
            num_occ_f -= np.sum(np.array([num_occ_fi[nif] for nif, str_f in
                                          enumerate(name_occ_fi) if str_f == "Va"]).astype(int))
    num_occ_f = int(num_occ_f)

    struc_screen_f = {}
    for nif_main in range(int(NSW_MC_G)):
        path_res_main = os.path.join(path_tmp, "tmp_screening")
        if not os.path.exists(path_res_main):
            os.mkdir(path_res_main)

        struc_screen_f[nif_main] = {}
        for nif_scr in range(int(SCQU1)):
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
            energy_system_f, struc_screen_i_f, _, _ = (
                cal_opt(Model_CHGNET, rm_dummy_atom(struc_f), path_res_sub, IBRION, EDIFFG2, NSW_OPT2, ISIF2, OFI_OPT2,
                        np.array([DOF_A, DOF_B, DOF_C, DOF_BC, DOF_AC, DOF_AB]),
                        True, True, True, True, True))
            energy_screen_mat_f[nif_main, nif_scr] = energy_system_f / Num_occ
            struc_screen_f[nif_main][nif_scr] = struc_screen_i_f
            cal_xrd(struc_screen_i_f, path_res_sub)

            gen(path_log,
                num_printer(nif_main + 1, 8, 0, 0) + " / " +
                num_printer(nif_scr + 1, 8, 0, 0) +
                num_printer(energy_screen_mat_f[nif_main, nif_scr], 24, 6, 0) +
                num_printer(nif_main + 1, 19 + NSW_MC_G_dight, 0, 0) + "-" +
                num_printer(nif_sub, NSW_MC_S_dight, 0, 0), True, True, True)

    gen(path_log, "", True, True, True)
    num_screened_f = min([int(SCQU2), int(NSW_MC_G * SCQU1)])
    energy_screened_f = np.zeros(num_screened_f)
    atom_doped_mat_screened_f = {}
    for nif_key, nif_str_array in FASEQ_array_index_dict.items():
        atom_doped_mat_screened_f[nif_key] = np.zeros((num_screened_f, nif_str_array.shape[0]), dtype=object)
    energy_screen_mat_min_f = np.min(energy_screen_mat_f)
    for nif_scr in range(energy_screened_f.shape[0]):
        energy_screened_f[nif_scr] = np.min(energy_screen_mat_f)

        index_min_f = np.argmin(energy_screen_mat_f)
        index_min_f = np.unravel_index(index_min_f, energy_screen_mat_f.shape)
        index_min_2nd_f = int(index_screen_mat_f[index_min_f[0], index_min_f[1]])
        for nif_key, nif_str_array in Atom_doped_mat.items():
            atom_doped_mat_screened_f[nif_key][nif_scr, :] = nif_str_array[int(index_min_f[0]), index_min_2nd_f, :]

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

    num_group_f = int(SCQU2)
    num_iteration_f = int(NSW_MCCE)
    if num_iteration_f == 0:
        return

    num_occ_f = np.sum(Atom_num)
    for nif_key, nif_str_array in FASEQ_array_index_dict.items():
        name_occ_fi, num_occ_fi = NAME_OCC_dict[nif_key], NUM_OCC_dict[nif_key]
        if "Va" in name_occ_fi:
            num_occ_f -= np.sum(np.array([num_occ_fi[nif] for nif, str_f in
                                          enumerate(name_occ_fi) if str_f == "Va"]).astype(int))
    num_occ_f = int(num_occ_f)
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
                "\n                               Iteration      Sampling type   Energy (eV/atoms)     "
                "dE (eV)    Acceptance rate     State\n", True, True, True)

        structure_save_f = {}

        # initialization
        path_res_sub = os.path.join(path_struc_opt, "tmp_" + format(nif_struc + 1, '.0f') + "_1")
        if not os.path.exists(path_res_sub):
            os.mkdir(path_res_sub)

        structure_f = struc_origin_f.copy()
        atom_doped_mat_old_f = {}
        for nif_key, nif_str_array in atom_doped_screened_mat_f.items():
            atom_doped_mat_f = nif_str_array[nif_struc, :]
            structure_f = replace_atom(structure_f, atom_doped_mat_f, FASEQ_array_index_dict[nif_key])
            atom_doped_mat_old_f[nif_key] = atom_doped_mat_f
        structure_clear_f = rm_dummy_atom(structure_f)
        hash_mat_f[nif_struc, 0] = get_structure_hash(structure_clear_f)

        atom_name_mc_f, atom_num_mc_f = wr_atom_list(structure_f.atomic_numbers)
        relax_atom_f = np.zeros(structure_f.frac_coords.shape) + 1
        if "selective_dynamics" in structure_f.site_properties:
            relax_atom_f = np.array(structure_f.site_properties["selective_dynamics"])
        wr_contcar(path_res_sub, "POSCAR", Title, structure_f.lattice.matrix, structure_f.frac_coords,
                   atom_name_mc_f, atom_num_mc_f, relax_atom_f, None)
        energy_system_f, struc_opted_f, _, _ = (
            cal_opt(Model_CHGNET, structure_clear_f, path_res_sub, IBRION, EDIFFG3, NSW_OPT3, ISIF3, OFI_OPT3,
                    np.array([DOF_A, DOF_B, DOF_C, DOF_BC, DOF_AC, DOF_AB]),
                    True, True, True, True, True))
        energy_mat_f[nif_struc, 0] = energy_system_f / num_occ_f
        structure_save_f[0] = struc_opted_f
        cal_xrd(struc_opted_f, path_res_sub)
        if num_iteration_f > 1:
            gen(path_log, num_printer(1, 18, 0, 0) +
                num_printer(energy_mat_f[nif_struc, 0], 32, 6, 0), True, True, True)

        structure_find_index_f = 0
        structure_accept_f = structure_f.copy()
        for nif_step in range(1, num_iteration_f):
            group_key_f = np.random.choice(list(atom_doped_screened_mat_f.keys()))
            atom_doped_mat_f = select_index(atom_doped_mat_old_f[group_key_f], FASEQ_array_index_dict[group_key_f])
            structure_f = replace_atom(structure_accept_f, atom_doped_mat_f, FASEQ_array_index_dict[group_key_f])
            structure_clear_f = rm_dummy_atom(structure_f)
            hash_v = get_structure_hash(structure_clear_f)
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
                           atom_name_mc_f, atom_num_mc_f, relax_atom_f, None)
                energy_system_f, struc_opted_f, _, _ = (
                    cal_opt(Model_CHGNET, structure_clear_f, path_res_sub, IBRION, EDIFFG3, NSW_OPT3, ISIF3, OFI_OPT3,
                            np.array([DOF_A, DOF_B, DOF_C, DOF_BC, DOF_AC, DOF_AB]),
                            True, True, True, True, True))
                energy_mat_f[nif_struc, nif_step] = energy_system_f / num_occ_f
                structure_save_f[nif_step] = struc_opted_f
                cal_xrd(struc_opted_f, path_res_sub)

                delta_e_f = energy_mat_f[nif_struc, nif_step] - energy_mat_f[nif_struc, structure_find_index_f]
                delta_e_joule = delta_e_f * num_occ_f * q_cons
                temp_mc_f = TEMP_MCI + (TEMP_MCF - TEMP_MCI) * nif_step / (num_iteration_f - 1)
                acceptance_rate_f = min([1, np.exp(-delta_e_joule / kB_cons / temp_mc_f)])
                if acceptance_rate_f >= np.random.rand():
                    state_mc_f = "Acceptance"
                    atom_doped_mat_old_f[group_key_f] = atom_doped_mat_f
                    structure_accept_f = structure_f.copy()
                    structure_find_index_f = nif_step
                else:
                    state_mc_f = "*Rejection"

                gen(path_log,
                    num_printer(nif_step + 1, 18, 0, 0) + MC_Key_dict[group_key_f] +
                    num_printer(energy_mat_f[nif_struc, nif_step], 20, 6, 0) +
                    num_printer(delta_e_f * num_occ_f, 16, 3, 0) +
                    num_printer(acceptance_rate_f * 100, 14, 2, 0) + "%              " +
                    state_mc_f, True, True, True)

        gen(path_log, "Statistic thermodynamic properties of Structure " +
            format(nif_struc + 1, '.0f') + " / " + format(num_group_f, '.0f') +
            ":\n                                   "
            "Steps       Progress      E_elec        U          S_vib        G       |  "
            "<E_elec>        <U>         <S_vib>      S_shannons       <S>         "
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
                           False, True, False,
                           True, True, True))

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


def cal_pre_lc_opt(path_sub, model_chgnet, lc_mat_f, dof_so, struc_f, coord_relax_f, ratio_f):
    wr_contcar(path_sub, "POSCAR", Title, lc_mat_f, struc_f.frac_coords,
               Atom_type, Atom_num, coord_relax_f, np.zeros((struc_f.frac_coords.shape[0], 3)))

    struc_partial_f = Structure(Lattice(lc_mat_f), struc_f.species, struc_f.frac_coords)
    lc_array_f = np.concatenate((Lattice(lc_mat_f).abc, Lattice(lc_mat_f).angles), axis=0)
    max_stress_f = np.zeros(2)  # normal and shear
    for nif_lc in range(int(NSW_SOLC / 2)):
        prediction_f = model_chgnet.predict_structure(struc_partial_f)
        lc_change_f = prediction_f["s"]
        lc_change_f += lc_change_f.T
        lc_change_f = np.array([lc_change_f[0, 0], lc_change_f[1, 1], lc_change_f[2, 2],
                                lc_change_f[1, 2], lc_change_f[0, 2], lc_change_f[0, 1]])
        max_stress_f = np.maximum(max_stress_f, np.array([np.max(np.abs(lc_change_f[: 3])),
                                                          np.max(np.abs(lc_change_f[3: ]))]))
        lc_change_f *= dof_so

        lc_array_f[: 3] *= 1 - ratio_f * RAT_SOLC * lc_change_f[: 3] / max_stress_f[0]
        lc_array_f[3: ] *= 1 - ratio_f * RAT_SOLC * lc_change_f[3: ] / max_stress_f[1]
        lc_mat_new_f = (
            Lattice.from_parameters(float(lc_array_f[0]), float(lc_array_f[1]), float(lc_array_f[2]),
                                    float(lc_array_f[3]), float(lc_array_f[4]), float(lc_array_f[5])))

        struc_partial_f = Structure(lc_mat_new_f, struc_f.species, struc_f.frac_coords)

    num_loop_f = max([int(NSW_SOLC) - int(NSW_SOLC / 2), 5])
    num_loop_array_f = np.arange(0, num_loop_f + 1, np.floor(num_loop_f / 5))
    num_loop_array_f[-1] = num_loop_f
    for nif_lc in range(num_loop_array_f.shape[0] - 1):
        _, struc_partial_opted_f, _, _ = (
            cal_opt(model_chgnet, struc_partial_f, path_sub, IBRION, EDIFFG,
                    int(num_loop_array_f[nif_lc + 1]) - int(num_loop_array_f[nif_lc]), ISIF, OFI_OPT, dof_so,
                    False, False, False, False, False))

        struc_partial_f = Structure(struc_partial_opted_f.lattice, struc_f.species, struc_f.frac_coords)

    wr_contcar(path_sub, "POSCAR_LC_pre-optimized", Title, struc_partial_f.lattice.matrix, struc_f.frac_coords,
               Atom_type, Atom_num, coord_relax_f, np.zeros((struc_f.frac_coords.shape[0], 3)))

    return struc_partial_f


def cal_constrained_opt(path_job, path_log, model_chgnet, struc_f, energy_origin_f):
    if len(DOF_SO1) != 0:
        if len(DOF_SO2) != 0:
            energy_mat_f = np.zeros((int(DOF_SON1), int(DOF_SON2)))
            stress_mat_f = np.zeros((int(DOF_SON1), int(DOF_SON2), 3, 3))
            lc_change_mat_f = np.zeros((int(DOF_SON1), int(DOF_SON2), 3, 3))
            ratio_modify_f = np.zeros((int(DOF_SON1), int(DOF_SON2)))
        else:
            energy_mat_f = np.zeros((int(DOF_SON1)))
            stress_mat_f = np.zeros((int(DOF_SON1), 3, 3))
            lc_change_mat_f = np.zeros((int(DOF_SON1), 1, 3, 3))
            ratio_modify_f = np.zeros((int(DOF_SON1)))
    else:
        if len(DOF_SO2) != 0:
            energy_mat_f = np.zeros((int(DOF_SON2)))
            stress_mat_f = np.zeros((int(DOF_SON2), 3, 3))
            lc_change_mat_f = np.zeros((1, int(DOF_SON2), 3, 3))
            ratio_modify_f = np.zeros((int(DOF_SON2)))
        else:
            gen(path_log, "The input does not set the strain and the program terminates.",
                True, True, True)
            exit(0)

    if bool(set(DOF_SO1) & set(DOF_SO2)):
        gen(path_log, "The entered strain setting is repeated and the program terminates.",
            True, True, True)
        exit(0)

    path_out = os.path.join(path_job, "CHGNet_results_Constrained_optimization.log")
    if (len(DOF_SO1) != 0) & (len(DOF_SO2) != 0):
        with open(path_out, "w") as nf_rf:
            nf_rf.write("# " + st_str + "\n\n      Strain_1st      Strain_2nd    Energy difference (eV)    "
                                        "Stress     A           B           C          "
                                        "B-C         A-C         A-B   (GPa)\n")
    else:
        with open(path_out, "w") as nf_rf:
            nf_rf.write("# " + st_str + "\n\n      Strain    Energy difference (eV)    "
                                        "Stress     A           B           C          "
                                        "B-C         A-C         A-B   (GPa)\n")

    sign_dof_f1 = np.array([1, 1, 1, 1, 1, 1])
    boolean_dof_f1 = np.array([False, False, False, False, False, False])
    for nif_s1 in DOF_SO1:
        nif_s1_merge = "".join(sorted(nif_s1.upper()))
        if nif_s1_merge[0] == "-":
            nif_s1_merge = nif_s1_merge[1:]
            if "A" == nif_s1_merge:
                sign_dof_f1[0] = -1
                boolean_dof_f1[0] = True
            if "B" == nif_s1_merge:
                sign_dof_f1[1] = -1
                boolean_dof_f1[1] = True
            if "C" == nif_s1_merge:
                sign_dof_f1[2] = -1
                boolean_dof_f1[2] = True
            if "BC" == nif_s1_merge:
                sign_dof_f1[3] = -1
                boolean_dof_f1[3] = True
            if "AC" == nif_s1_merge:
                sign_dof_f1[4] = -1
                boolean_dof_f1[4] = True
            if "AB" == nif_s1_merge:
                sign_dof_f1[5] = -1
                boolean_dof_f1[5] = True
        else:
            if "A" == nif_s1_merge:
                boolean_dof_f1[0] = True
            if "B" == nif_s1_merge:
                boolean_dof_f1[1] = True
            if "C" == nif_s1_merge:
                boolean_dof_f1[2] = True
            if "BC" == nif_s1_merge:
                boolean_dof_f1[3] = True
            if "AC" == nif_s1_merge:
                boolean_dof_f1[4] = True
            if "AB" == nif_s1_merge:
                boolean_dof_f1[5] = True

    sign_dof_f2 = np.array([1, 1, 1, 1, 1, 1])
    boolean_dof_f2 = np.array([False, False, False, False, False, False])
    for nif_s2 in DOF_SO2:
        nif_s2_merge = "".join(sorted(nif_s2.upper()))
        if nif_s2_merge[0] == "-":
            nif_s2_merge = nif_s2_merge[1:]
            if "A" == nif_s2_merge:
                sign_dof_f2[0] = -1
                boolean_dof_f2[0] = True
            if "B" == nif_s2_merge:
                sign_dof_f2[1] = -1
                boolean_dof_f2[1] = True
            if "C" == nif_s2_merge:
                sign_dof_f2[2] = -1
                boolean_dof_f2[2] = True
            if "BC" == nif_s2_merge:
                sign_dof_f2[3] = -1
                boolean_dof_f2[3] = True
            if "AC" == nif_s2_merge:
                sign_dof_f2[4] = -1
                boolean_dof_f2[4] = True
            if "AB" == nif_s2_merge:
                sign_dof_f2[5] = -1
                boolean_dof_f2[5] = True
        else:
            if "A" == nif_s2_merge:
                boolean_dof_f2[0] = True
            if "B" == nif_s2_merge:
                boolean_dof_f2[1] = True
            if "C" == nif_s2_merge:
                boolean_dof_f2[2] = True
            if "BC" == nif_s2_merge:
                boolean_dof_f2[3] = True
            if "AC" == nif_s2_merge:
                boolean_dof_f2[4] = True
            if "AB" == nif_s2_merge:
                boolean_dof_f2[5] = True

    lc_array_origin_f = np.concatenate((struc_f.lattice.abc, struc_f.lattice.angles), axis=0)
    if lc_change_mat_f.shape[0] == 1:  # for (1, N)
        for nif_so2 in range(lc_change_mat_f.shape[1]):
            lc_array_f2 = lc_array_origin_f.copy()
            for nif_s2 in range(6):
                if boolean_dof_f2[nif_s2]:
                    if sign_dof_f1[nif_s2] == 1:
                        rat_f = DOF_SOL2 + (DOF_SOH2 - DOF_SOL2) * nif_so2 / (lc_change_mat_f.shape[1] - 1)
                    else:
                        rat_f = DOF_SOH2 - (DOF_SOH2 - DOF_SOL2) * nif_so2 / (lc_change_mat_f.shape[1] - 1)

                    lc_array_f2[nif_s2] *= rat_f
                    ratio_modify_f[nif_so2] += (1 - rat_f) ** 2

            lc_change_mat_f[0, nif_so2, :, :] = (
                Lattice.from_parameters(float(lc_array_f2[0]), float(lc_array_f2[1]), float(lc_array_f2[2]),
                                        float(lc_array_f2[3]), float(lc_array_f2[4]), float(lc_array_f2[5])).matrix)
    else:  # for (M, 1) or (M, N)
        for nif_so1 in range(lc_change_mat_f.shape[0]):
            lc_array_f1 = lc_array_origin_f.copy()
            for nif_s1 in range(6):
                if boolean_dof_f1[nif_s1]:
                    if sign_dof_f1[nif_s1] == 1:
                        rat_f = DOF_SOL1 + (DOF_SOH1 - DOF_SOL1) * nif_so1 / (lc_change_mat_f.shape[0] - 1)
                    else:
                        rat_f = DOF_SOH1 - (DOF_SOH1 - DOF_SOL1) * nif_so1 / (lc_change_mat_f.shape[0] - 1)

                    lc_array_f1[nif_s1] *= rat_f

                    if lc_change_mat_f.shape[1] == 1:  # for (M, 1)
                        ratio_modify_f[nif_so1] += (1 - rat_f) ** 2
                    else:  # for (M, N)
                        ratio_modify_f[nif_so1, :] += (1 - rat_f) ** 2

            if lc_change_mat_f.shape[1] == 1:  # for (M, 1)
                lc_change_mat_f[nif_so1, 0, :, :] = (
                    Lattice.from_parameters(float(lc_array_f1[0]), float(lc_array_f1[1]), float(lc_array_f1[2]),
                                            float(lc_array_f1[3]), float(lc_array_f1[4]), float(lc_array_f1[5])).matrix)
            else:  # for (M, N)
                for nif_so2 in range(lc_change_mat_f.shape[1]):
                    lc_array_f2 = lc_array_f1.copy()
                    for nif_s2 in range(6):
                        if boolean_dof_f2[nif_s2]:
                            if sign_dof_f2[nif_s2]:
                                rat_f = DOF_SOL2 + (DOF_SOH2 - DOF_SOL2) * nif_so2 / (lc_change_mat_f.shape[1] - 1)
                            else:
                                rat_f = DOF_SOH2 - (DOF_SOH2 - DOF_SOL2) * nif_so2 / (lc_change_mat_f.shape[1] - 1)

                            lc_array_f2[nif_s2] *= rat_f
                            ratio_modify_f[nif_so1, nif_so2] += (1 - rat_f) ** 2

                    lc_change_mat_f[nif_so1, nif_so2, :, :] = (
                        Lattice.from_parameters(float(lc_array_f2[0]), float(lc_array_f2[1]),
                                                float(lc_array_f2[2]), float(lc_array_f2[3]),
                                                float(lc_array_f2[4]), float(lc_array_f2[5])).matrix)

    dof_so = (boolean_dof_f1 + boolean_dof_f2) == False
    dof_so *= np.array([DOF_A, DOF_B, DOF_C, DOF_BC, DOF_AC, DOF_AB])
    dof_st_f = np.array(["A", "B", "C", "BC", "AC", "AB"], dtype=object)

    path_tmp = os.path.join(path_job, "CHGNet_tmp")
    if not os.path.exists(path_tmp):
        os.mkdir(path_tmp)

    if (lc_change_mat_f.shape[0] == 1) or (lc_change_mat_f.shape[1] == 1):
        if lc_change_mat_f.shape[0] == 1:
            num_file_f = lc_change_mat_f.shape[1]
            lc_change_mat_f = lc_change_mat_f[0, :, :, :]
        else:
            num_file_f = lc_change_mat_f.shape[0]
            lc_change_mat_f = lc_change_mat_f[:, 0, :, :]

        for nif_file in range(num_file_f):
            path_sub = os.path.join(path_tmp, "tmp_" + format(nif_file + 1, '.0f'))
            if not os.path.exists(path_sub):
                os.mkdir(path_sub)

            gen(path_log, "Current process:   ", True, True, False)

            if "selective_dynamics" in struc_f.site_properties:
                coord_relax_f = np.array(struc_f.site_properties["selective_dynamics"])
            else:
                coord_relax_f = np.zeros(struc_f.frac_coords.shape[0]) + 1

            struc_partial_f = cal_pre_lc_opt(path_sub, model_chgnet, lc_change_mat_f[nif_file, :, :],
                                             dof_so, struc_f, coord_relax_f, 1 - np.exp(-10 * ratio_modify_f[nif_file]))
            gen(path_log, "(Lattice optimization completed)    ", True, False, False)

            if "selective_dynamics" in struc_f.site_properties:
                struc_partial_f.add_site_property("selective_dynamics",
                                                  struc_f.site_properties["selective_dynamics"])

            energy_f, struc_partial_opted_f, conv_boolean_f, stress_f = (
                cal_opt(model_chgnet, struc_partial_f, path_sub, IBRION, EDIFFG, NSW_OPT, ISIF, OFI_OPT, dof_so,
                        True, True, False, True, True))
            cal_xrd(struc_partial_opted_f, path_sub)
            wr_xdatcar_head(path_job, "Trajectory_VASP_Constrained_optimization",
                            lc_change_mat_f[nif_file, :, :], Title, Atom_type, Atom_num)
            wr_xdatcar_coord(path_job, "Trajectory_VASP_Constrained_optimization", nif_file, struc_partial_opted_f.frac_coords)
            energy_mat_f[nif_file] = energy_f
            stress_mat_f[nif_file, :, :] = np.array([[stress_f[0], stress_f[5], stress_f[4]],
                                                     [stress_f[5], stress_f[1], stress_f[3]],
                                                     [stress_f[4], stress_f[3], stress_f[2]]])

            if conv_boolean_f:
                gen(path_log, format(nif_file + 1, '.0f') + " / " + format(num_file_f, '.0f') +
                    "    => Reach convergence", True, False, True)
            else:
                gen(path_log, format(nif_file + 1, '.0f') + " / " + format(num_file_f, '.0f') +
                    "    => Not yet converged", True, False, True)

            with open(path_out, "a") as nf_rf:
                if len(DOF_SO1) != 0:
                    nf_rf.write(num_printer(DOF_SOL1 + nif_file / (num_file_f - 1) *
                                            (DOF_SOH1 - DOF_SOL1), 12, 4, 0))
                else:
                    nf_rf.write(num_printer(DOF_SOL2 + nif_file / (num_file_f - 1) *
                                            (DOF_SOH2 - DOF_SOL2), 12, 4, 0))
                nf_rf.write(num_printer(energy_f - energy_origin_f, 17, 4, 0) +
                            num_printer(stress_f[0], 28, 4, 0) +
                            num_printer(stress_f[1], 12, 4, 0) +
                            num_printer(stress_f[2], 12, 4, 0) +
                            num_printer(stress_f[3], 12, 4, 0) +
                            num_printer(stress_f[4], 12, 4, 0) +
                            num_printer(stress_f[5], 12, 4, 0) + "\n")

        if lc_change_mat_f.shape[0] == 1:  # 2nd
            xp_f = np.linspace(DOF_SOL2, DOF_SOH2, num_file_f)
            st_xlabel_f = " & ".join(dof_st_f[boolean_dof_f2].tolist())
        else:
            xp_f = np.linspace(DOF_SOL1, DOF_SOH1, num_file_f)
            st_xlabel_f = " & ".join(dof_st_f[boolean_dof_f1].tolist())
        plot_f(xp_f, energy_mat_f - energy_origin_f, "Relative energy (eV)", "r",
               "Strain of " + st_xlabel_f, "Energy (eV)",
               "Energy vs. strain", os.path.join(path_job, "_Energy_scan_LC"), 0)

        fig_f, axes_f = plt.subplots(3, 1, figsize=(16, 9))
        for nif_sub in range(3):
            axes_f[nif_sub].plot(xp_f, stress_mat_f[:, nif_sub, nif_sub], color="k")
            axes_f[nif_sub].set_ylabel("Stress " + dof_st_f[nif_sub] + " (GPa)",
                                       fontsize=16, color="k", fontweight="bold")
            axes_f[nif_sub].grid(True, linestyle="--")
        axes_f[0].set_title("Normal stress", fontsize=20, color="k", fontweight="bold")
        axes_f[2].set_xlabel("Normal strain", fontsize=20, color="k", fontweight="bold")
        plt.savefig(os.path.join(path_job, "_Normal stress_scan_LC.png"))
        plt.close()

        fig_f, axes_f = plt.subplots(3, 1, figsize=(16, 9))
        axes_f[0].plot(xp_f, stress_mat_f[:, 1, 2], color="b")
        axes_f[0].set_ylabel("Stress B-C (GPa)", fontsize=16, color="k", fontweight="bold")
        axes_f[0].grid(True, linestyle="--")
        axes_f[1].plot(xp_f, stress_mat_f[:, 0, 2], color="b")
        axes_f[1].set_ylabel("Stress A-C (GPa)", fontsize=16, color="k", fontweight="bold")
        axes_f[1].grid(True, linestyle="--")
        axes_f[2].plot(xp_f, stress_mat_f[:, 0, 1], color="b")
        axes_f[2].set_ylabel("Stress A-B (GPa)", fontsize=16, color="k", fontweight="bold")
        axes_f[2].grid(True, linestyle="--")
        axes_f[0].set_title("Shear stress", fontsize=20, color="k", fontweight="bold")
        axes_f[2].set_xlabel("Shear strain", fontsize=20, color="k", fontweight="bold")
        plt.savefig(os.path.join(path_job, "_Shear stress_scan_LC.png"))
        plt.close()
    else:
        for nif_folder in range(lc_change_mat_f.shape[0]):
            path_folder = os.path.join(path_tmp, "tmp_" + format(nif_folder + 1, '.0f'))
            if not os.path.exists(path_folder):
                os.mkdir(path_folder)

            for nif_file in range(lc_change_mat_f.shape[1]):
                path_sub = os.path.join(path_folder, "tmp_" + format(nif_folder + 1, '.0f') +
                                        "-" + format(nif_file + 1, '.0f'))
                if not os.path.exists(path_sub):
                    os.mkdir(path_sub)

                gen(path_log, "Current process:   ", True, True, False)

                if "selective_dynamics" in struc_f.site_properties:
                    coord_relax_f = np.array(struc_f.site_properties["selective_dynamics"])
                else:
                    coord_relax_f = np.zeros(struc_f.frac_coords.shape[0]) + 1

                struc_partial_f = (
                    cal_pre_lc_opt(path_sub, model_chgnet, lc_change_mat_f[nif_folder, nif_file, :, :],
                                   dof_so, struc_f, coord_relax_f,
                                   1 - np.exp(-10 * ratio_modify_f[nif_folder, nif_file])))
                gen(path_log, "(Lattice optimization completed)    ", True, False, False)

                if "selective_dynamics" in struc_f.site_properties:
                    struc_partial_f.add_site_property("selective_dynamics",
                                                      struc_f.site_properties["selective_dynamics"])

                energy_f, struc_partial_opted_f, conv_boolean_f, stress_f = (
                    cal_opt(model_chgnet, struc_partial_f, path_sub, IBRION, EDIFFG, NSW_OPT, ISIF, OFI_OPT, dof_so,
                            True, True, False, True, True))
                wr_xdatcar_head(path_job, "Trajectory_VASP_Constrained_optimization",
                                lc_change_mat_f[nif_folder, nif_file, :, :], Title, Atom_type, Atom_num)
                wr_xdatcar_coord(path_job, "Trajectory_VASP_Constrained_optimization",
                                 nif_folder * lc_change_mat_f.shape[1] + nif_file,
                                 struc_partial_opted_f.frac_coords)
                cal_xrd(struc_partial_opted_f, path_sub)

                energy_mat_f[nif_folder, nif_file] = energy_f
                stress_mat_f[nif_folder, nif_file, :, :] = np.array([[stress_f[0], stress_f[5], stress_f[4]],
                                                                     [stress_f[5], stress_f[1], stress_f[3]],
                                                                     [stress_f[4], stress_f[3], stress_f[2]]])

                with open(path_out, "a") as nf_rf:
                    nf_rf.write(num_printer(DOF_SOL1 + nif_folder / (lc_change_mat_f.shape[0] - 1) *
                                            (DOF_SOH1 - DOF_SOL1), 14, 4, 0) +
                                num_printer(DOF_SOL2 + nif_file / (lc_change_mat_f.shape[1] - 1) *
                                            (DOF_SOH2 - DOF_SOL2), 16, 4, 0) +
                                num_printer(energy_f - energy_origin_f, 19, 4, 0) +
                                num_printer(stress_f[0], 28, 4, 0) +
                                num_printer(stress_f[1], 12, 4, 0) +
                                num_printer(stress_f[2], 12, 4, 0) +
                                num_printer(stress_f[3], 12, 4, 0) +
                                num_printer(stress_f[4], 12, 4, 0) +
                                num_printer(stress_f[5], 12, 4, 0) + "\n")

                if conv_boolean_f:
                    gen(path_log, format(nif_file + 1, '.0f') + " / " + format(DOF_SON2, '.0f') + "      " +
                        format(nif_folder + 1, '.0f') + " / " + format(DOF_SON1, '.0f') +
                        "    => Reach convergence", True, False, True)
                else:
                    gen(path_log, format(nif_file + 1, '.0f') + " / " + format(DOF_SON2, '.0f') + "      " +
                        format(nif_folder + 1, '.0f') + " / " + format(DOF_SON1, '.0f') +
                        "    => Not yet converged", True, False, True)

            gen(path_log, "", True, False, True)

            if nif_folder > 0:
                so1f, so2f = np.meshgrid(np.linspace(DOF_SOL1, DOF_SOH1, int(DOF_SON1)),
                                         np.linspace(DOF_SOL2, DOF_SOH2, int(DOF_SON2)), indexing='ij')
                energy_z = energy_mat_f - np.min(energy_mat_f)
                plt.figure(figsize=(16, 9), dpi=300)
                plt_so = plt.contourf(so1f[: nif_folder + 1, :], so2f[: nif_folder + 1, :],
                                      energy_z[: nif_folder + 1, :], cmap=cmap_linear,
                                      levels=np.linspace(0, np.max(energy_z[: nif_folder + 1, :]), 101))
                cbar = plt.colorbar(plt_so)
                cbar.set_label("Relative energy (eV)", fontsize=16)
                plt_so_line = (plt.contour(so1f[: nif_folder + 1, :], so2f[: nif_folder + 1, :],
                                           energy_z[: nif_folder + 1, :],
                                           np.arange(0, np.max(energy_z[: nif_folder + 1, :]) * 1.001,
                                                     np.round(np.max(energy_z[: nif_folder + 1, :]) / 8)),
                                           colors='black', linestyles="--"))
                plt.clabel(plt_so_line, inline=True, fontsize=12, fmt='%.0f')
                plt.grid(True, linewidth=0.5, alpha=0.7)
                st_xlabel_f = " & ".join(dof_st_f[boolean_dof_f1].tolist())
                plt.xlabel("Strain of " + st_xlabel_f, fontsize=20, color="k", fontweight="bold")
                st_ylabel_f = " & ".join(dof_st_f[boolean_dof_f2].tolist())
                plt.ylabel("Strain of " + st_ylabel_f, fontsize=20, color="k", fontweight="bold")
                plt.savefig(os.path.join(path_job, "_Energy_scan_LC.png"))
                plt.close()

                for nif_plt in range(6):
                    if nif_plt == 0:
                        file_name_f, data_f = "_Normal stress A_scan_LC.png", stress_mat_f[: nif_folder + 1, :, 0, 0]
                    elif nif_plt == 1:
                        file_name_f, data_f = "_Normal stress B_scan_LC.png", stress_mat_f[: nif_folder + 1, :, 1, 1]
                    elif nif_plt == 2:
                        file_name_f, data_f = "_Normal stress C_scan_LC.png", stress_mat_f[: nif_folder + 1, :, 2, 2]
                    elif nif_plt == 3:
                        file_name_f, data_f = "_Shear stress B-C_scan_LC.png", stress_mat_f[: nif_folder + 1, :, 1, 2]
                    elif nif_plt == 4:
                        file_name_f, data_f = "_Shear stress A-C_scan_LC.png", stress_mat_f[: nif_folder + 1, :, 0, 2]
                    elif nif_plt == 5:
                        file_name_f, data_f = "_Shear stress A-B_scan_LC.png", stress_mat_f[: nif_folder + 1, :, 0, 1]

                    plt.figure(figsize=(16, 9), dpi=300)
                    if (data_f.min() < 0) and (data_f.max() > 0):
                        norm_f = mcolors.TwoSlopeNorm(vmin=data_f.min(), vcenter=0, vmax=data_f.max())
                        plt_so = plt.contourf(so1f[: nif_folder + 1, :], so2f[: nif_folder + 1, :], data_f,
                                              cmap="jet", norm=norm_f,
                                              levels=np.linspace(np.min(data_f), np.max(data_f), 101))
                    else:
                        plt_so = plt.contourf(so1f[: nif_folder + 1, :], so2f[: nif_folder + 1, :], data_f,
                                              cmap="jet", levels=np.linspace(np.min(data_f), np.max(data_f), 101))
                    cbar = plt.colorbar(plt_so)
                    plt_so_line = (plt.contour(so1f[: nif_folder + 1, :], so2f[: nif_folder + 1, :], data_f,
                                               np.arange(np.min(data_f), np.max(data_f) * 1.001,
                                                         np.ceil((np.max(data_f) - np.min(data_f)) / 8)),
                                               colors='black', linestyles="--"))
                    plt.clabel(plt_so_line, inline=True, fontsize=12, fmt='%.0f')
                    cbar.set_label("Stress (GPa)", fontsize=16)
                    plt.grid(True, linewidth=0.5, alpha=0.7)
                    plt.xlabel("Strain of " + st_xlabel_f, fontsize=20, color="k", fontweight="bold")
                    plt.ylabel("Strain of " + st_ylabel_f, fontsize=20, color="k", fontweight="bold")
                    plt.savefig(os.path.join(path_job, file_name_f))
                    plt.close()


def cal_neb_pre_process(path_job, path_log, model_chgnet, ibroin_f, images_f):
    if (ibroin_f == 80) or (ibroin_f == 83):
        gen(path_log, 'Reading completed, calculation method is "Nudged Elastic Band"',
            True, True, True)
    elif (ibroin_f == 90) or (ibroin_f == 93):
        gen(path_log, 'Reading completed, calculation method is "Climbing-Image Nudged Elastic Band"',
            True, True, True)

    if images_f is None:
        images_f = 0
        while True:
            path_sub_pos = os.path.join(path_job, format(images_f, '02d'), "POSCAR")
            if os.path.exists(path_sub_pos):
                images_f += 1
            else:
                break
        if images_f <= 2:
            gen(path_log, "The program has not read "
                          "the complete file and has terminated.", True, True, True)
            exit(0)
        else:
            images_f -= 2
    else:
        for nif_file in range(images_f + 2):
            path_sub_pos = os.path.join(path_job, format(nif_file, '02d'), "POSCAR")
            if not os.path.exists(path_sub_pos):
                gen(path_log, "The program has not read "
                              "the complete file and has terminated.", True, True, True)
                exit(0)

    gen(path_log, "The program has read " + format(images_f, '.0f') +
        " files to be optimized and continues to calculate...", True, True, True)

    _, atom_type_f, atom_num_f, lc_f, coord_car_f, coord_relax_old_f, struc_f = rd_pos(
        os.path.join(Path_job, "00", "POSCAR"))
    atom_list_f = [site.species_string for site in struc_f]
    struc_mat_f = np.zeros((int(images_f) + 2, int(np.sum(atom_num_f)), 3))
    struc_mat_f[0, :, :] = coord_car_f
    lc_mat_f = lc_f.copy()
    energy_mat_f = np.zeros((NSW_T, int(images_f) + 2))
    for ni_file in range(1, int(images_f) + 2):
        _, _, _, lc_f, coord_car_f, coord_relax_f, struc_f = (
            rd_pos(os.path.join(Path_job, format(ni_file, '02d'), "POSCAR")))

        if np.std(lc_mat_f - lc_f) >= 1E-8:
            gen(Path_log, "If the grid constants are different, the program terminates.",
                True, True, True)
            exit(0)
        if not atom_list_f == [site.species_string for site in struc_f]:
            gen(Path_log, "The atom types or order are inconsistent, the program terminates.",
                True, True, True)
            exit(0)
        if np.sum(coord_relax_f != coord_relax_old_f) != 0:
            gen(Path_log, "In file " + format(ni_file, '02d') +
                ", the atomic relaxation type was found to be inconsistent, and the program terminated.",
                True, True, True)
            exit(0)

        diff_match_f = coord_car_f - struc_mat_f[ni_file - 1, :, :]
        diff_match_f = diff_match_f[:, np.newaxis, :] + np.dot(D_or, lc_f)[np.newaxis, :, :]
        diff_match_value_f = np.einsum('ijk, ijk -> ij', diff_match_f, diff_match_f, optimize=True)
        struc_mat_f[ni_file, :, :] = (struc_mat_f[ni_file - 1, :, :] +
                                      diff_match_f[np.arange(diff_match_value_f.shape[0]),
                                      np.argmin(diff_match_value_f, axis=1), :])

    struc_i_f = Structure(lattice=Lattice(lc_mat_f, pbc=(PBC_A, PBC_B, PBC_C)), species=atom_list_f,
                          coords=np.dot(struc_mat_f[0, :, :], np.linalg.inv(lc_mat_f)))
    energy_mat_f[:, 0] = model_chgnet.predict_structure(struc_i_f)["e"]
    struc_f_f = Structure(lattice=Lattice(lc_mat_f, pbc=(PBC_A, PBC_B, PBC_C)), species=atom_list_f,
                          coords=np.dot(struc_mat_f[-1, :, :], np.linalg.inv(lc_mat_f)))
    energy_mat_f[:, -1] = model_chgnet.predict_structure(struc_f_f)["e"]

    gen(Path_log, "", True, True, True)
    gen(Path_log, '        Step      Energy difference   '
                  'from "00"   from "' + format(images_f + 1, '02d') +
        '"            Image        Force_max \n'
        '                                                 '
        '(eV)                                       (floder name)  (eV/Angstrom)',
        True, True, True)

    st = "# " + st_str + ("\n\n"
                          "                  Energy (eV/atoms)\n"
                          "                  Image\n"
                          "   Step")
    for nif_image in range(int(images_f) + 2):
        st += "             " + format(nif_image, '02d')
    with open(Path_store, 'w') as nf_f:
        nf_f.write(st + "\n")

    return int(images_f), atom_type_f, atom_num_f, atom_list_f, lc_mat_f, struc_mat_f, coord_relax_f, energy_mat_f


def wr_plt_neb(path_job, nif_step, struc_mat_f, disp, force_opt, energy_mat_f):
    path_xdatcar = os.path.join(path_job, "Trajectory_VASP")
    if os.path.exists(path_xdatcar):
        os.remove(path_xdatcar)
    wr_xdatcar_head(path_job, LC_mat, "NEB_images", Atom_type, Atom_num)
    wr_xdatcar_coord(path_job, 0, np.dot(struc_mat_f[0, :, :], np.linalg.inv(LC_mat)))
    struc_mat_f[1: -1, :, :] += disp
    for ni_image in range(1, IMAGES_int + 2):  # rearrange adjacent images
        diff_match = struc_mat_f[ni_image, :, :] - struc_mat_f[ni_image - 1, :, :]
        diff_match = diff_match[:, np.newaxis, :] + np.dot(D_or, LC_mat)[np.newaxis, :, :]
        diff_match_value = np.einsum('ijk, ijk -> ij', diff_match, diff_match, optimize=True)
        struc_mat_f[ni_image, :, :] = (struc_mat_f[ni_image - 1, :, :] +
                                       diff_match[np.arange(diff_match_value.shape[0]),
                                       np.argmin(diff_match_value, axis=1), :])
        coord_frac = np.dot(struc_mat_f[ni_image, :, :], np.linalg.inv(LC_mat))
        wr_contcar(os.path.join(path_job, format(ni_image, '02d')), "CONTCAR", "NEB_images",
                   LC_mat, coord_frac, Atom_type, Atom_num, Coord_relax, None)
        wr_xdatcar_coord(path_job, ni_image, coord_frac)

    force_max = np.sqrt(np.max(np.einsum('ijk, ijk -> ij',
                                         force_opt, force_opt, optimize=True), axis=1))
    gen(Path_log, num_printer(nif_step + 1, 12, 0, 0) +
        num_printer((np.max(energy_mat_f[nif_step, :]) - energy_mat_f[nif_step, 0]) *
                    np.sum(Atom_num), 35, 4, 0) +
        num_printer((np.max(energy_mat_f[nif_step, :]) - energy_mat_f[nif_step, -1]) *
                    np.sum(Atom_num), 12, 4, 0) +
        "              " + format(np.argmax(energy_mat_f[nif_step, :]), '02d') +
        num_printer(np.max(force_max), 17, 3, 0), True, True, True)

    st = num_printer(nif_step + 1, 8, 0, 0) + "    "
    for ni_image in range(IMAGES_int + 2):
        st += num_printer(energy_mat_f[nif_step, ni_image], 15, 8, 0)
    gen(Path_store, st, False, False, True)

    diff_energy = (energy_mat_f[nif_step, :] - energy_mat_f[nif_step, 0]) * np.sum(Atom_num)
    fig, ax1 = plt.subplots(figsize=(16, 9), dpi=200)
    ax1.plot(np.arange(0, IMAGES_int + 2, 1), diff_energy,
             label="Energy difference (eV)", color="r", marker='o', linewidth=2)
    ax1.set_xlabel("Images", fontsize=20, color="k", fontweight="bold")
    ax1.set_ylabel("Energy difference (eV)", fontsize=20, color="r", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=16, loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, IMAGES_int + 1, 1), force_max, label="Force_max (eV/Angstrom)", color="b", linewidth=1)
    ax2.set_ylabel("Force_max (eV/Angstrom)", fontsize=20, color="b", fontweight="bold")
    plt.title("Energy & Force vs. Images", fontsize=24, fontweight="bold")
    plt.legend(fontsize=16, loc='upper right')
    plt.savefig(os.path.join(path_job, "_CHGNet_NEB_Energy_Force.png"))
    plt.close()

    if nif_step > 1:
        image_f, step_f = np.meshgrid(np.arange(0, IMAGES_int + 2, 1), np.arange(0, NSW_T + 1, 1))  # for plot
        energy_z = energy_mat_f[: nif_step + 1, :] - energy_mat_f[0, 0]
        energy_z *= np.sum(Atom_num)

        plt.figure(figsize=(16, 9), dpi=200)
        plt_neb = (
            plt.contourf(image_f[: nif_step + 1, :], step_f[: nif_step + 1, :], energy_z,
                         cmap=cmap_linear, levels=np.linspace(np.min(energy_z), np.max(energy_z), 101)))
        cbar = plt.colorbar(plt_neb)
        cbar.set_label("Relative energy (eV)", fontsize=16)
        plt_neb_line = (
            plt.contour(image_f[: nif_step + 1, :], step_f[: nif_step + 1, :], energy_z,
                        np.arange(np.min(energy_z), np.max(energy_z) + 0.2, 0.2),
                        colors='black', linestyles="--"))
        plt.clabel(plt_neb_line, inline=True, fontsize=8, fmt='%.1f')
        plt.grid(True, linewidth=0.5, alpha=0.7)
        plt.xlabel("Images", fontsize=20, color="k", fontweight="bold")
        plt.ylabel("Steps", fontsize=20, color="k", fontweight="bold")

        plt.savefig(os.path.join(path_job, "_CHGNet_NEB_3D_plot.png"))
        plt.close()

    return force_max


def cal_neb(path_job, model_chgnet, struc_mat_f, energy_mat_f):
    velocity_fire = np.zeros((IMAGES_int, int(np.sum(Atom_num)), 3))
    delta_t_fire, delta_t_fire_max = np.zeros((IMAGES_int, int(np.sum(Atom_num)))) + 2E-16, 1E-15  # in sec
    delta_t_fire_r_inc, delta_t_fire_r_dec = 1.1, 0.5
    alpha_fire_initial, alpha_fire_decay = 0.2, 0.99
    num_step_use = min([NSW_T / 10, 5])
    fire_use_mat = np.zeros((IMAGES_int, int(np.sum(Atom_num))))
    mass_fire = 10  # in amu
    for nif_step in range(NSW_T):  # for each iteration
        force_real = np.zeros((IMAGES_int, int(np.sum(Atom_num)), 3))
        tang_vec_unit = np.zeros((IMAGES_int, int(np.sum(Atom_num)), 3))
        force_struc_orth = np.zeros((IMAGES_int, int(np.sum(Atom_num)), 3))
        for ni_seq in range(1, IMAGES_int + 1):  # for each image
            struc_image_f = (
                Structure(lattice=Lattice(LC_mat, pbc=(PBC_A, PBC_B, PBC_C)), species=Atom_list,
                          coords=np.dot(struc_mat_f[ni_seq, :, :], np.linalg.inv(LC_mat))))

            struc_infor = model_chgnet.predict_structure(struc_image_f)
            energy_mat_f[nif_step, ni_seq], force_real[ni_seq - 1, :, :] = struc_infor["e"], struc_infor["f"]
            tang_vec = struc_mat_f[ni_seq + 1, :, :] - struc_mat_f[ni_seq - 1, :, :]
            tang_vec_length = np.sqrt(np.einsum('ij, ij', tang_vec, tang_vec, optimize=True))
            tang_vec_unit[ni_seq - 1, :, :] = tang_vec / max([tang_vec_length, 1E-12])
            dot_product_unit = (np.einsum('ij, ij -> i', force_real[ni_seq - 1, :, :],
                                          tang_vec_unit[ni_seq - 1, :, :], optimize=True))
            force_struc_orth[ni_seq - 1, :, :] = force_real[ni_seq - 1, :, :] - dot_product_unit[:,
                                                                                np.newaxis] * tang_vec

        spring_cons_array = (SPRING_S + (SPRING_L - SPRING_S) *
                             (energy_mat_f[nif_step, 1: -1] - np.min(energy_mat_f[nif_step, 1: -1])) /
                             (np.max(energy_mat_f[nif_step, 1: -1]) - np.min(energy_mat_f[nif_step, 1: -1])))

        dis_image = np.zeros(IMAGES_int + 1)
        for ni_dis in range(IMAGES_int + 1):
            diff_image = struc_mat_f[ni_dis + 1, :, :] - struc_mat_f[ni_dis, :, :]
            diff_image = np.sqrt(np.einsum('ij, ij -> i', diff_image, diff_image, optimize=True))
            dis_image[ni_dis] = np.sum(diff_image)

        force_spring = spring_cons_array * (dis_image[1:] - dis_image[: -1])  # in eV/Angstrom
        force_spring = force_spring[:, np.newaxis, np.newaxis] * tang_vec_unit

        if IBRION // 10 == 8:  # NEB
            force_opt = force_struc_orth + force_spring
        elif IBRION // 10 == 9:  # CI-NEB
            force_opt = force_struc_orth + force_spring
            index_climb = np.argmax(energy_mat_f[nif_step, 1: -1])
            force_opt[index_climb] = 2 * force_struc_orth[index_climb] - force_real[index_climb]
        force_opt[:, Coord_relax == 0] = 0

        if (IBRION == 80) or (IBRION == 90):  # use steepest descent
            disp = force_opt / np.max(np.abs(force_opt)) * POTIM_T
        elif (IBRION == 83) or (IBRION == 93):  # use FIRE
            dot_p = np.einsum('ijk, ijk -> ij', force_opt, velocity_fire, optimize=True)
            fire_use_mat[dot_p >= 0] += 1
            fire_use_mat[dot_p < 0] = 0
            alpha_fire = alpha_fire_initial * alpha_fire_decay ** np.maximum(np.zeros(dot_p.shape),
                                                                             fire_use_mat - num_step_use)
            delta_t_fire[(dot_p >= 0) * (fire_use_mat > num_step_use)] *= delta_t_fire_r_inc
            delta_t_fire[(dot_p < 0)] *= delta_t_fire_r_dec
            delta_t_fire = np.minimum(delta_t_fire, delta_t_fire_max + np.zeros(dot_p.shape))

            velocity_fire += (force_opt / mass_fire * q_cons * NA_cons * 1E23 *
                              delta_t_fire[:, :, np.newaxis])  # in Angstrom/s**2
            velocity_fire_force_contri = (
                    np.einsum('ijk, ijk -> ij', velocity_fire, velocity_fire, optimize=True) /
                    (np.einsum('ijk, ijk -> ij', force_opt, force_opt, optimize=True) + 1E-24))
            velocity_fire_force_contri = np.sqrt(velocity_fire_force_contri)[:, :, np.newaxis] * force_opt
            velocity_fire = ((1 - alpha_fire[:, :, np.newaxis]) * velocity_fire +
                             alpha_fire[:, :, np.newaxis] * velocity_fire_force_contri)
            velocity_fire[dot_p < 0, :] = 0

            disp = velocity_fire * delta_t_fire[:, :, np.newaxis]
            # disp *= POTIM_T / np.max(np.abs(disp))

        force_max = wr_plt_neb(path_job, nif_step, struc_mat_f, disp, force_opt, energy_mat_f)

        if all(force_max <= EDIFFG_T):
            gen(Path_log, "Reaching convergence conditions.", True, True, True)

            if IBRION // 10 == 9:
                gen(Path_log, "", True, True, True)
                gen(Path_log, "CI-NEB calculation is completed and phonon calculation is performed...",
                    True, True, True)
                index_image_highest = np.argmax(energy_mat_f[nif_step, :])
                struc_image_f = (
                    Structure(lattice=Lattice(LC_mat, pbc=(PBC_A, PBC_B, PBC_C)), species=Atom_list,
                              coords=np.dot(struc_mat_f[index_image_highest, :, :], np.linalg.inv(LC_mat))))
                cal_phonon(path_job, model_chgnet, struc_image_f, TEMP_STA, True, True,
                           True, True, True, True)
            exit(0)


# Start point
if __name__ == "__main__":
    st_time = time.time()
    st_str = "Made by Zhong-Lun Li, Dec. 17 2025"
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
                      'and continue running using the previously trained model...\n', True, True, True)
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
    XRD_FWHM = 0.2

    EDIFFG = 1E-2
    NSW_OPT = 500
    OFI_OPT = 1
    ISIF = True
    DOF_A, DOF_B, DOF_C = True, True, True
    DOF_BC, DOF_AC, DOF_AB = True, True, True
    NSW_SOLC = 100
    RAT_SOLC = 1E-2
    DOF_SO1, DOF_SO2 = [], []
    DOF_SOL1, DOF_SOH1, DOF_SON1 = 0.8, 1.2, 10
    DOF_SOL2, DOF_SOH2, DOF_SON2 = 0.8, 1.2, 10

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

    FASEQ, NAME_OCC_dict, NUM_OCC_dict = {}, {}, {}
    MIN_R_MC = None
    NSW_MC_G = 1
    NSW_MC_S = 1E3
    EDIFFG0 = 1E-5
    TEMP_MCI = 298
    TEMP_MCF = 298
    SCQU1 = 10
    EDIFFG1 = 0.05
    NSW_OPT1 = 50
    OFI_OPT1 = 1
    ISIF1 = True
    NUM_CONF = 50
    E_SACONF = 2
    SCQU2 = 3
    EDIFFG2 = 0.01
    NSW_OPT2 = 500
    OFI_OPT2 = 1
    ISIF2 = True
    NSW_MCCE = 0
    E_MCCE = 1
    EDIFFG3 = 0.01
    NSW_OPT3 = 100
    OFI_OPT3 = 1
    ISIF3 = True
    POTIM_CE = 0.01
    CUT_WNCE = 50
    TEMP_CE = 298
    CVG_S_CE = 1E-6

    EDIFFG_M = 1E-2
    NSW_MP = 100
    OFI_MP = 1
    RATIO_MP = 0.01

    EDIFFG_T = 1E-2
    POTIM_T = 0.1
    NSW_T = 100
    IMAGES = None
    SPRING_S, SPRING_L = 5, 5

    MIL_IND = []
    NUM_TB = 2
    NUM_FB = 1
    H_VAC = 10
    EDIFFG_I = 0.05
    NSW_I = 50
    OFI_I = 1
    NUM_CS = 10
    EDIFFG_F = 0.01
    NSW_F = 500
    OFI_F = 1

    if os.path.exists(os.path.join(Path_job, "Input_CHGNet")):
        rd_cncar(os.path.join(Path_job, "Input_CHGNet"))
    else:
        wr_cncar()

    if (ENSB == "npt") & (THEMT == "Berendsen"):
        THEMT = "Berendsen_inhomogeneous"

    Num_MC_group = len(FASEQ)
    if MIN_R_MC is None:
        MIN_R_MC = 1 / (Num_MC_group + 1)
    else:
        MIN_R_MC = min([MIN_R_MC, 1 / min([1, Num_MC_group])])
    MC_sampling_num = dict.fromkeys(FASEQ.keys(), 0)

    Path_pos = os.path.join(Path_job, "POSCAR")
    if (IBRION != 80) and (IBRION != 83) and (IBRION != 90) and (IBRION != 93):
        if not os.path.exists(Path_pos):
            gen(Path_log, 'The "POSCAR" does not exist '
                          'and the program terminates abnormally.\n', True, True, True)
            exit(0)
        Title, Atom_type, Atom_num, LC, Coord, Coord_relax, STRUCture = rd_pos(Path_pos)

    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.filterwarnings("ignore", message="logm result may be inaccurate")

    if IBRION == -1:
        gen(Path_log, 'Reading completed, calculation method is "Single point calculation".\n'
                      '                           '
                      'Steps      Volume             E0               dE             '
                      'Force_max         Magnetic moment\n                                   '
                      '(Angstrom^3)      (eV/atoms)       (eV/atoms)       (eV/Angstrom)          (mu_B)\n', True, True,
            True)
        cal_xrd(STRUCture, Path_job)
        cal_sp(Model_CHGNET, 0, 0)

    elif IBRION == 0:
        gen(Path_log, 'Reading completed, calculation method is "Molecular Dynamics".\n'
                      '                               '
                      'Time     Temperature     Volume          E0           dE           E0_K         dE_K          '
                      'E0_P         dE_P       Force_max      Magnetic moment\n'
                      '                               (ps)         (K)      (Angstrom^3)   '
                      '   (eV)         (eV)          (eV)         (eV)          (eV)      '
                      '   (eV)     (eV/Angstrom)       (mu_B)\n', True, True, True)

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
        cal_xrd(STRUCture, Path_job)
        (Num_real_large_freq, Num_real_low_freq, Num_image_freq,
         Vib_freq, Vib_mode, Vib_ZPE, Vib_Internal_energy, Vib_Entropy) = (
            cal_phonon(Path_job, Model_CHGNET, STRUCture, TEMP_STA,
                       False, True, True,
                       True, True, True))

    elif IBRION % 10 == 5:
        if IBRION // 10 == 1:
            IBRION = "LBFGS"
        elif IBRION // 10 == 3:
            IBRION = "FIRE"

        _, STRUCture, _, _ = (
            cal_opt(Model_CHGNET, STRUCture, Path_job, IBRION, EDIFFG, NSW_OPT, ISIF, OFI_OPT,
                    np.array([DOF_A, DOF_B, DOF_C, DOF_BC, DOF_AC, DOF_AB]),
                    True, True, True, True, True))

        gen(Path_log, "", True, True, True)
        gen(Path_log, "Structural optimization is completed and phonon calculation is performed...",
            True, True, True)
        cal_phonon(Path_job, Model_CHGNET, STRUCture, TEMP_STA, False, True, True,
                   True, True, True)
        cal_xrd(STRUCture, Path_job)

    elif (IBRION == 71) or (IBRION == 73):
        if IBRION % 10 == 1:
            IBRION = "LBFGS"
        elif IBRION % 10 == 3:
            IBRION = "FIRE"

        Path_tmp = os.path.join(Path_job, "CHGNet_tmp")
        if not os.path.exists(Path_tmp):
            os.mkdir(Path_tmp)

        cal_xrd(STRUCture, Path_job)
        Elas_mat_abc, Elas_mat_xyz = cal_elas_tensor(Model_CHGNET, STRUCture, RATIO_MP, Path_tmp)
        wr_mech_prop(Path_log, Elas_mat_abc, Elas_mat_xyz)

    elif (IBRION == 1) or (IBRION == 3):
        if IBRION % 10 == 1:
            IBRION = "LBFGS"
        elif IBRION % 10 == 3:
            IBRION = "FIRE"

        _, STRUCture, _, _ = (
            cal_opt(Model_CHGNET, STRUCture, Path_job, IBRION, EDIFFG, NSW_OPT, ISIF, OFI_OPT,
                    np.array([DOF_A, DOF_B, DOF_C, DOF_BC, DOF_AC, DOF_AB]),
                    True, True, True, True, True))
        cal_xrd(STRUCture, Path_job)

    elif (IBRION == 11) or (IBRION == 13):
        if IBRION % 10 == 1:
            IBRION = "LBFGS"
        elif IBRION % 10 == 3:
            IBRION = "FIRE"

        gen(Path_log, 'Reading completed, calculation method is "Sampling Optimization".',
            True, True, True)

        FASEQ_array, FASEQ_array_index_dict = {}, {}
        for ni_key in FASEQ.keys():
            FASEQ_array[ni_key] = np.zeros(STRUCture.num_sites)
            for ni in range(len(FASEQ[ni_key])):
                data_i = FASEQ[ni_key][ni].split(":")
                if len(data_i) == 3:
                    FASEQ_array[ni_key][int(data_i[0]): int(data_i[2]) + 1: int(data_i[1])] = True
                elif len(data_i) == 2:
                    FASEQ_array[ni_key][int(data_i[0]): int(data_i[1]) + 1] = True
                elif len(data_i) == 1:
                    FASEQ_array[ni_key][int(data_i[0])] = True
            FASEQ_array_index_dict[ni_key] = np.where(FASEQ_array[ni_key])[0]

            if len(NAME_OCC_dict[ni_key]) != len(NUM_OCC_dict[ni_key]):
                gen(Path_log, 'Error! The lengths of the atom type label "' + " ".join(NAME_OCC_dict[ni_key]) +
                    '" and the value label "' + " ".join(NUM_OCC_dict[ni_key]) + '" do not match! Program aborted.\n',
                    True, True, True)
                exit(0)

        NSW_MC_G_dight = np.ceil(np.log10(NSW_MC_G) + 1)
        NSW_MC_S_dight = np.ceil(np.log10(NSW_MC_S) + 1)
        STRUCture_site_prop = STRUCture.site_properties
        Path_tmp = os.path.join(Path_job, "CHGNet_tmp")
        if not os.path.exists(Path_tmp):
            os.mkdir(Path_tmp)

        MC_Key_dict = {}
        for ni_key in FASEQ.keys():
            st_key = ""
            for ni_st in range(11 - len(ni_key)):
                st_key += " "
            st_key += ni_key
            MC_Key_dict[ni_key] = st_key
        # 1st screening
        Num_occ, Energy_save_mat, STRUCture_screen, Atom_doped_mat = cal_mc(Path_log, STRUCture)
        # 2nd screening
        Energy_screened, Atom_doped_screened_mat = cal_mc_opt(Path_log, Path_tmp)
        # Configuration calculation using Monte Carlo method
        cal_mcce(Path_log, Path_tmp, Atom_doped_screened_mat, STRUCture)
        
        Path_rm = os.path.join(Path_job, "POSCAR_current")
        if os.path.exists(Path_rm):
            os.remove(Path_rm)
        Path_rm = os.path.join(Path_job, "CONTCAR_current")
        if os.path.exists(Path_rm):
            os.remove(Path_rm)

    elif (IBRION == 21) or (IBRION == 23):
        if IBRION % 10 == 1:
            IBRION = "LBFGS"
        elif IBRION % 10 == 3:
            IBRION = "FIRE"

        from matplotlib.colors import LinearSegmentedColormap

        colors_linear = [(0.0, "blue"), (0.2, "palegreen"), (0.3, "yellow"), (0.5, "darkorange"), (1.0, "red")]
        cmap_linear = LinearSegmentedColormap.from_list("cmap_linear", colors_linear, N=101)

        Energy_opted, STRUCture_opted, _, _ = (
            cal_opt(Model_CHGNET, STRUCture, Path_job, IBRION, EDIFFG, NSW_OPT, ISIF, OFI_OPT,
                    np.array([DOF_A, DOF_B, DOF_C, DOF_BC, DOF_AC, DOF_AB]),
                    True, False, True, True, True))
        cal_xrd(STRUCture, Path_job)

        gen(Path_log, "", True, False, True)
        gen(Path_log, 'Reading completed, calculation method is "Constrained Optimization".\n',
            True, True, True)
        cal_constrained_opt(Path_job, Path_log, Model_CHGNET, STRUCture_opted, Energy_opted)

    elif (IBRION == 80) or (IBRION == 83) or (IBRION == 90) or (IBRION == 93):
        from matplotlib.colors import LinearSegmentedColormap

        colors_linear = [(0.0, "lightskyblue"), (0.2, "palegreen"), (0.3, "yellow"), (0.5, "darkorange"), (1.0, "red")]
        cmap_linear = LinearSegmentedColormap.from_list("cmap_linear", colors_linear, N=101)

        Path_store = os.path.join(Path_job, "_CHGNet_NEB_Energy_data.log")
        X, Y, Z = np.meshgrid(np.arange(-PBC_A, PBC_A + 1),
                              np.arange(-PBC_B, PBC_B + 1), np.arange(-PBC_C, PBC_C + 1))
        D_or = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

        IMAGES_int, Atom_type, Atom_num, Atom_list, LC_mat, STRUCture_mat, Coord_relax, Energy_mat = (
            cal_neb_pre_process(Path_job, Path_log, Model_CHGNET, IBRION, IMAGES))

        cal_neb(Path_job, Model_CHGNET, STRUCture_mat, Energy_mat)

    elif (IBRION == 101) or (IBRION == 103):
        if IBRION % 10 == 1:
            IBRION = "LBFGS"
        elif IBRION % 10 == 3:
            IBRION = "FIRE"

        Path_tmp = os.path.join(Path_job, "CHGNet_tmp")
        if not os.path.exists(Path_tmp):
            os.mkdir(Path_tmp)

        # optimize the bulk
        Path_sub = os.path.join(Path_tmp, "Screen_bulk")
        if not os.path.exists(Path_sub):
            os.mkdir(Path_sub)
        Energy_bulk, STRUCture_opted, _, _ = (
            cal_opt(Model_CHGNET, STRUCture, Path_sub, IBRION, EDIFFG, NSW_OPT, ISIF, OFI_OPT,
                    np.array([DOF_A, DOF_B, DOF_C, DOF_BC, DOF_AC, DOF_AB]),
                    True, True, True, True, True))
        cal_xrd(STRUCture_opted, Path_sub)
        gen(Path_log, "Bulk, the energy is " + format(Energy_bulk / np.sum(Atom_num), '.3f') + " eV/atoms",
            0, True, True)
        gen(Path_log, "", 0, True, True)

        STRUCture_opted = SpacegroupAnalyzer(STRUCture_opted).get_conventional_standard_structure()
        Slab = SlabGenerator(STRUCture_opted, miller_index=tuple(MIL_IND.astype(int)),
                             in_unit_planes=True, min_slab_size=NUM_TB, min_vacuum_size=1,
                             reorient_lattice=True, center_slab=False).get_slabs()
        Path_sub = os.path.join(Path_tmp, "Screen_1st")
        if not os.path.exists(Path_sub):
            os.mkdir(Path_sub)
        Slab_1st_screen, Slab_2nd_screen = {}, {}
        Energy_mat_1st, Energy_mat_2nd = np.zeros(len(Slab)), np.zeros(int(NUM_CS))
        PBC_A = True
        PBC_B = True
        PBC_C = False
        # for 1st stage
        for ni_seq, ni_struc_ori in enumerate(Slab):
            Path_sub_n = os.path.join(Path_sub, "Termination_" + format(ni_seq + 1, '.0f'))
            if not os.path.exists(Path_sub_n):
                os.mkdir(Path_sub_n)

            ni_struc = ni_struc_ori.get_orthogonal_c_slab()
            LC_mat_slab = ni_struc.lattice.matrix.copy()
            Atom_type_slab, Atom_num_slab = wr_atom_list(ni_struc.atomic_numbers)
            Cart_coord_slab = ni_struc.cart_coords
            Cart_coord_slab -= Cart_coord_slab[np.argmin(Cart_coord_slab[:, -1]), :][np.newaxis, :]

            LC_mat_slab[-1, -1] = np.max(Cart_coord_slab[:, -1]) + H_VAC
            Frac_coord_slab = np.dot(Cart_coord_slab, np.linalg.inv(LC_mat_slab))
            Constraint_atom_mat = np.zeros(Frac_coord_slab.shape)
            for nii_seq, nii_name in enumerate(np.unique(Atom_type_slab)):
                Height_array = np.zeros(int(np.sum(Atom_num_slab))) * 1E10
                Num_atom_select = 0
                for ni3_seq in np.where(Atom_type_slab == nii_name)[0]:
                    st_atom_seq = int(np.sum(Atom_num_slab[: ni3_seq]))
                    ed_atom_seq = int(np.sum(Atom_num_slab[: ni3_seq + 1]))
                    Num_atom_select += Atom_num_slab[ni3_seq]
                    Height_array[st_atom_seq: ed_atom_seq] = Frac_coord_slab[st_atom_seq: ed_atom_seq, -1]

                Sort_atom = np.argsort(np.argsort(Height_array))
                Sort_atom_boolean = Sort_atom >= Num_atom_select * NUM_FB / NUM_TB
                for ni3_seq in np.where(Atom_type_slab == nii_name)[0]:
                    st_atom_seq = int(np.sum(Atom_num_slab[: ni3_seq]))
                    ed_atom_seq = int(np.sum(Atom_num_slab[: ni3_seq + 1]))
                    Constraint_atom_mat[st_atom_seq: ed_atom_seq, :] = Sort_atom_boolean[st_atom_seq:
                                                                                         ed_atom_seq][:, np.newaxis]

            wr_contcar(Path_sub_n, "POSCAR", Title, LC_mat_slab, Frac_coord_slab,
                       Atom_type_slab, Atom_num_slab, Constraint_atom_mat, None)

            Struc_slab = ase_io.read(os.path.join(Path_sub_n, "POSCAR"))
            Struc_slab.pbc = [PBC_A, PBC_B, PBC_C]
            Struc_slab = AseAtomsAdaptor.get_structure(Struc_slab)
            Energy_mat_1st[ni_seq], STRUCture_opted_slab, _, _ = (
                cal_opt(Model_CHGNET, Struc_slab, Path_sub_n, IBRION, EDIFFG_I, NSW_I, False, OFI_I,
                        np.array([DOF_A, DOF_B, DOF_C, DOF_BC, DOF_AC, DOF_AB]),
                        True, True, True, True, True))
            Slab_1st_screen[ni_seq] = STRUCture_opted_slab

            gen(Path_log, "Termination " + format(ni_seq + 1, '.0f') + " / " + format(len(Slab), '.0f') +
                ", the energy is " + format(Energy_mat_1st[ni_seq] / Frac_coord_slab.shape[0], '.3f') + " eV/atoms",
                0, True, True)

        gen(Path_log, "", 0, True, True)
        Path_sub = os.path.join(Path_tmp, "Screen_2nd")
        if not os.path.exists(Path_sub):
            os.mkdir(Path_sub)
        Select_index = np.argsort(np.argsort(Energy_mat_1st))
        # for 2nd stage
        for ni_seq in range(int(NUM_CS)):
            struc_index = np.where(Select_index == ni_seq)[0][0]
            Path_sub_n = os.path.join(Path_sub, "Termination_" + format(ni_seq + 1, '.0f'))
            if not os.path.exists(Path_sub_n):
                os.mkdir(Path_sub_n)

            Energy_mat_2nd[ni_seq], STRUCture_opted_slab, _, _ = (
                cal_opt(Model_CHGNET, Slab_1st_screen[struc_index], Path_sub_n, IBRION, EDIFFG_F, NSW_F, False, OFI_F,
                        np.array([DOF_A, DOF_B, DOF_C, DOF_BC, DOF_AC, DOF_AB]),
                        True, True, True, True, True))
            Slab_2nd_screen[ni_seq] = STRUCture_opted_slab

            gen(Path_log, "Termination " + format(ni_seq + 1, '.0f') + " / " + format(NUM_CS, '.0f') +
                ", the energy is " + format(Energy_mat_2nd[ni_seq] / Frac_coord_slab.shape[0], '.3f') + " eV/atoms",
                0, True, True)

        # surface energy calculation
        Surface_energy = ((Energy_mat_2nd - Energy_bulk * Frac_coord_slab.shape[0] / np.sum(Atom_num))
                          / (2 * np.linalg.det(LC_mat_slab[:2, :2])) * 1000)
        Surface_energy_sort = np.argsort(np.argsort(Surface_energy))
        for ni_seq in range(int(NUM_CS)):
            Struc_index = np.where(Surface_energy_sort == ni_seq)[0][0]
            shutil.copy(os.path.join(Path_sub, "Termination_" + format(Struc_index + 1, '.0f'), "CONTCAR"),
                        os.path.join(Path_job, "CONTCAR_" + format(ni_seq + 1, '.0f') + "_" +
                                     format(Surface_energy[Struc_index], '.0f') + "_meVperAngstromsq"))


    gen(Path_log, "", True, True, True)
    gen(Path_log, "Program terminated! This took " +
        num_printer((time.time() - st_time) / 3600, 8, 2, 0) + " hours.",
        True, True, True)

