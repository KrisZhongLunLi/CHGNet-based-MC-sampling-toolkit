# Modify by Zhong-Lun Li Sep. 16 2025,
# Department of Chemical Engineering, National Taiwan University of Science and Technology, Taipei 106, Taiwan

# The CHGNet model comes from this study:
# Deng, B., Zhong, P., Jun, K. et al. CHGNet as a pretrained universal neural network potential for
# charge-informed atomistic modelling. Nat Mach Intell 5, 1031–1041 (2023). https://doi.org/10.1038/s42256-023-00716-3


import os, glob, torch, time, sys, io, random
import numpy as np
from chgnet.utils import read_json
from matplotlib.lines import lineStyles
from pymatgen.core import Structure
from chgnet.data.dataset import StructureData
from multiprocessing import Process
from datetime import datetime

from requests.packages import target
from sympy.physics.units import force
from typing import Optional
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from chgnet.graph import CrystalGraph


def rd_cncar_t(path_f):
    with open(path_f, 'r', encoding="utf-8") as nf_f:
        data_f = nf_f.read()
        data_f = data_f.split("\n")
    for nif in range(len(data_f)):
        data_nif = data_f[nif]
        if "#" in data_f[nif]:
            data_nif = data_nif[:data_nif.index("#")]
        data_nif = data_nif.split()
        if len(data_nif) == 0:
            continue

        if data_nif[0] == "RAT_TRA":
            global RAT_TRA
            RAT_TRA = float(data_nif[-1])
        elif data_nif[0] == "RAT_VAL":
            global RAT_VAL
            RAT_VAL = float(data_nif[-1])
        elif data_nif[0] == "LEA_R":
            global LEA_R
            LEA_R = float(data_nif[-1])
        elif data_nif[0] == "CRT":
            global CRT
            if int(data_nif[-1]) == 0:
                CRT = "MAE"
            elif int(data_nif[-1]) == 1:
                CRT = "MSE"
            elif int(data_nif[-1]) == 2:
                CRT = "Huber"
        elif data_nif[0] == "LRS":
            global LRS
            if int(data_nif[-1]) == 0:
                LRS = "CosLR"
            elif int(data_nif[-1]) == 1:
                LRS = "ExponentialLR"
        elif data_nif[0] == "EPOCH":
            global EPOCH
            EPOCH = int(data_nif[-1])
        elif data_nif[0] == "FPTO":
            global FPTO
            FPTO = int(data_nif[-1])
        elif data_nif[0] == "BATCH":
            global BATCH
            BATCH = int(data_nif[-1])
        elif data_nif[0] == "RANDSEED":
            global RANDSEED
            RANDSEED = int(data_nif[-1])


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


def gen(path_f, st_inp):  # Generate a text string and log it to the output file
    st_new = "[" + str(datetime.now().year) + "-" + "%02d" % datetime.now().month + "-" + \
             "%02d" % datetime.now().day + "] [" + "%02d" % datetime.now().hour + ":" + \
             "%02d" % datetime.now().minute + ":" + "%02d" % datetime.now().second + "]  " + st_inp
    #print(st_new)
    with open(path_f, 'a', encoding="utf-8") as wr_ff:
        wr_ff.write(st_new + "\n")


def wr_cncar_t():
    gen(Path_log, '"Input_CHGNet_Training" does not exist. The program will automatically generate it.')
    gen(Path_log, "Please change the settings and resubmit the job.")
    with open(os.path.join(Path_job, "Input_CHGNet_Training"), 'w', encoding="utf-8") as nf_rf:
        nf_rf.write("# " + st_str + "\n\n"
                    "# General\n"
                    "#  Tags      Options   Default  Comments\n"
                    "  RAT_TRA  =  0.80     # [0.80] # the proportion of input used as training set\n"
                    "  RAT_VAL  =  0.10     # [0.10] # the proportion of input used as validation set\n\n"
                    "# Training part\n"
                    "  LEA_R    =  1E-2     # [1E-2] # learning rate\n"
                    "  CRT      =   1       # [1]    # types of loss functions, MAE (0), MSE (1) or Huber (2)\n"
                    "  LRS      =   1       # [1]    # learning rate scheduler, CosLR (1) or ExponentialLR (2)\n"
                    "  EPOCH    =  50       # [50]   # number of epochs for training\n"
                    "  FPTO     =  100      # [100]  # frequency to print training output\n"
                    "  BATCH    =   6       # [NCPUs]# the number of images calculated per synchronization\n"
                    "  RANDSEED =  42       # [None] # integer, random seeds are used to "
                                    "allocate training, validation, and test sets\n\n")

    exit(0)


def collate_graphs(batch_data: list) -> tuple[list[CrystalGraph], dict[str, torch.Tensor]]:
    """Collate of list of (graph, target) into batch data.

    Args:
        batch_data (list): list of (graph, target(dict))

    Returns:
        graphs (List): a list of graphs
        targets (Dict): dictionary of targets, where key and values are:
            e (Tensor): energies of the structures [batch_size]
            f (Tensor): forces of the structures [n_batch_atoms, 3]
            s (Tensor): stresses of the structures [3*batch_size, 3]
            m (Tensor): magmom of the structures [n_batch_atoms]
    """
    graphs = [graph for graph, _ in batch_data]
    all_targets = {key: [] for key in batch_data[0][1]}
    all_targets["e"] = torch.tensor([targets["e"] for _, targets in batch_data], dtype=torch.float32)

    for _, targets in batch_data:
        for target, value in targets.items():
            if target != "e":
                all_targets[target].append(value)

    return graphs, all_targets


def get_train_val_test_loader(
    *, dataset,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = True,
    rand_seed: Optional[int] = None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Randomly partition a dataset into train, val, test loaders.

    Args:
        dataset (Dataset): The dataset to partition.
        batch_size (int): The batch size for the data loaders
            Default = 64
        train_ratio (float): The ratio of the dataset to use for training
            Default = 0.8
        val_ratio (float): The ratio of the dataset to use for validation
            Default: 0.1
        num_workers (int): The number of worker processes for loading the data
            see torch Dataloader documentation for more info
            Default = 0
        pin_memory (bool): Whether to pin the memory of the data loaders
            Default: True
        rand_seed: Random seeds are used to allocate training, validation, and test sets

    Returns:
        train_loader, val_loader and optionally test_loader
    """
    total_size = len(dataset)
    indices = list(range(total_size))
    random.seed(rand_seed)
    random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=SubsetRandomSampler(indices=indices[0:train_size]),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=SubsetRandomSampler(
            indices=indices[train_size : train_size + val_size]
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=SubsetRandomSampler(indices=indices[train_size + val_size :]),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def train_main_sub(path_job, train_loader, val_loader, test_loader, model_chgnet,
                   training_target, lrs, crt, epoch, lea_r, fpto):
    from chgnet.trainer import Trainer
    # Define Trainer
    trainer = Trainer(model=model_chgnet, targets=training_target, optimizer="Adam", scheduler=lrs,
                      criterion=crt, epochs=epoch, learning_rate=lea_r, use_device="cpu", print_freq=fpto)
    trainer.train(train_loader, val_loader, test_loader, save_dir=path_job,  save_test_result=True)


def train_main(func, path_out, *args):
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)

    with open(path_out, "w", encoding="utf-8") as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)

        try:
            result = func(*args)
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)

    return result


def train_record(path_job, path_log, path_out, st_str_f, epoch_f, training_target, interval=10):
    time.sleep(interval)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.ioff()

    num_line_data, current_epoch, current_epoch_old = 2, 0, 0
    tra_loop, val_loop = 0, 0
    loss_record_f = np.zeros((2, epoch_f + 1, 5))  # [tra or val; EPOCH; loss, e, f, s, m]
    exit_loop = False
    while not exit_loop:
        with open(path_out, 'r', encoding="utf-8") as nf_rf:
            data_all = nf_rf.read()
            data_all = data_all.split("\n")
        if len(data_all) > num_line_data:
            data_new = data_all[num_line_data: -1]
            for nif in range(len(data_new)):
                data_pf = data_new[nif]
                if ("Epoch" not in data_pf) & ("Test" not in data_pf):
                    data_pf = data_pf[4:]
                if "*   " in data_pf:
                    gen(path_log, "")
                    continue
                if "Evaluate Model on Test Set" in data_pf:
                    gen(path_log, "")
                gen(path_log, data_pf)


                # plot
                data_pf = data_new[nif]
                if tra_loop == 0:
                    if "Epoch" in data_pf:
                        tra_loop = data_pf.split("/")[1]
                        tra_loop = int(tra_loop.split("]")[0])
                if val_loop == 0:
                    if "Val" in data_pf:
                        val_loop = data_pf.split("/")[1]
                        val_loop = int(val_loop.split("]")[0])

                if "[" + str(tra_loop) + "/" + str(tra_loop) + "]"in data_pf:
                    if "Epoch" in data_pf:
                        loss_record_f[0, current_epoch, 0] = float(data_pf.split()[7].split("(")[-1][:-1])
                        loss_record_f[0, current_epoch, 1] = float(data_pf.split()[11].split("(")[-1][:-1])
                        loss_record_f[0, current_epoch, 2] = float(data_pf.split()[13].split("(")[-1][:-1])
                        if "s" in training_target:
                            loss_record_f[0, current_epoch, 3] = float(data_pf.split()[15].split("(")[-1][:-1])
                        if "m" in training_target:
                            loss_record_f[0, current_epoch, 4] = float(data_pf.split()[17].split("(")[-1][:-1])
                if "[" + str(val_loop) + "/" + str(val_loop) + "]"in data_pf:
                    if "Val" in data_pf:
                        loss_record_f[1, current_epoch + 1, 0] = float(data_pf.split()[7].split("(")[-1][:-1])
                        loss_record_f[1, current_epoch + 1, 1] = float(data_pf.split()[11].split("(")[-1][:-1])
                        loss_record_f[1, current_epoch + 1, 2] = float(data_pf.split()[13].split("(")[-1][:-1])
                        if "s" in training_target:
                            loss_record_f[1, current_epoch + 1, 3] = float(data_pf.split()[15].split("(")[-1][:-1])
                        if "m" in training_target:
                            loss_record_f[1, current_epoch + 1, 4] = float(data_pf.split()[17].split("(")[-1][:-1])
                        current_epoch += 1

                if current_epoch <= 1:
                    continue
                if current_epoch_old == current_epoch:
                    continue

                # All targets
                current_epoch_old += 1
                st_f = "# " + st_str_f + "\n\n      Training     Validation   loss of all targets\n"
                for nif2 in range(current_epoch):
                    st_f += (num_printer(loss_record_f[0, nif2, 0], 14, 6, 0) +
                             num_printer(loss_record_f[1, nif2, 0], 14, 6, 0) + "\n")
                with open(os.path.join(path_job, "_CHGNET_Loss_All.log"), 'w') as nf_rf:
                    nf_rf.write(st_f)
                plt.subplots(figsize=(16, 9), dpi=300)
                plt.semilogy(np.arange(0, current_epoch, 1),
                         loss_record_f[0, :current_epoch, 0], color="r", label="Training")
                plt.semilogy(np.arange(1, current_epoch, 1),
                         loss_record_f[1, 1:current_epoch, 0], color="b", label="Validation")
                plt.xlabel("Training steps", fontsize=20)
                plt.ylabel("Loss of all targets", fontsize=20)
                plt.legend(fontsize=18)
                plt.grid(True, linestyle="--", linewidth=1)
                plt.savefig(os.path.join(path_job, "_CHGNET_Loss_All.png"))
                plt.close()

                # Energy
                st_f = "# " + st_str_f + "\n\n      Training     Validation   loss of energy\n"
                for nif2 in range(current_epoch):
                    st_f += (num_printer(loss_record_f[0, nif2, 1], 14, 6, 0) +
                             num_printer(loss_record_f[1, nif2, 1], 14, 6, 0) + "\n")
                with open(os.path.join(path_job, "_CHGNET_Loss_Energy.log"), 'w') as nf_rf:
                    nf_rf.write(st_f)
                plt.subplots(figsize=(16, 9), dpi=300)
                plt.semilogy(np.arange(0, current_epoch, 1),
                         loss_record_f[0, :current_epoch, 1], color="r", label="Training")
                plt.semilogy(np.arange(1, current_epoch, 1),
                         loss_record_f[1, 1:current_epoch, 1], color="b", label="Validation")
                plt.xlabel("Training steps", fontsize=20)
                plt.ylabel("Loss of energy", fontsize=20)
                plt.legend(fontsize=18)
                plt.grid(True, linestyle="--", linewidth=1)
                plt.savefig(os.path.join(path_job, "_CHGNET_Loss_Energy.png"))
                plt.close()

                # Force
                st_f = "# " + st_str_f + "\n\n      Training     Validation   loss of force\n"
                for nif2 in range(current_epoch):
                    st_f += (num_printer(loss_record_f[0, nif2, 2], 14, 6, 0) +
                             num_printer(loss_record_f[1, nif2, 2], 14, 6, 0) + "\n")
                with open(os.path.join(path_job, "_CHGNET_Loss_Force.log"), 'w') as nf_rf:
                    nf_rf.write(st_f)
                plt.subplots(figsize=(16, 9), dpi=300)
                plt.semilogy(np.arange(0, current_epoch, 1),
                         loss_record_f[0, :current_epoch, 2], color="r", label="Training")
                plt.semilogy(np.arange(1, current_epoch, 1),
                         loss_record_f[1, 1:current_epoch, 2], color="b", label="Validation")
                plt.xlabel("Training steps", fontsize=20)
                plt.ylabel("Loss of force", fontsize=20)
                plt.legend(fontsize=18)
                plt.grid(True, linestyle="--", linewidth=1)
                plt.savefig(os.path.join(path_job, "_CHGNET_Loss_Force.png"))
                plt.close()

                # Stress
                if "s" in training_target:
                    st_f = "# " + st_str_f + "\n\n      Training     Validation   loss of hydrostatic stress\n"
                    for nif2 in range(current_epoch):
                        st_f += (num_printer(loss_record_f[0, nif2, 3], 14, 6, 0) +
                                 num_printer(loss_record_f[1, nif2, 3], 14, 6, 0) + "\n")
                    with open(os.path.join(path_job, "_CHGNET_Loss_Hydrostatic stress.log"), 'w') as nf_rf:
                        nf_rf.write(st_f)
                    plt.subplots(figsize=(16, 9), dpi=300)
                    plt.semilogy(np.arange(0, current_epoch, 1),
                                 loss_record_f[0, :current_epoch, 3], color="r", label="Training")
                    plt.semilogy(np.arange(1, current_epoch, 1),
                                 loss_record_f[1, 1:current_epoch, 3], color="b", label="Validation")
                    plt.xlabel("Training steps", fontsize=20)
                    plt.ylabel("Loss of hydrostatic stress", fontsize=20)
                    plt.legend(fontsize=18)
                    plt.grid(True, linestyle="--", linewidth=1)
                    plt.savefig(os.path.join(path_job, "_CHGNET_Loss_Hydrostatic stress.png"))
                    plt.close()

                # Magnetic moment
                if "m" in training_target:
                    st_f = "# " + st_str_f + "\n\n      Training     Validation   loss of magnetic moment\n"
                    for nif2 in range(current_epoch):
                        st_f += (num_printer(loss_record_f[0, nif2, 4], 14, 6, 0) +
                                 num_printer(loss_record_f[1, nif2, 4], 14, 6, 0) + "\n")
                    with open(os.path.join(path_job, "_CHGNET_Loss_Magnetic moment.log"), 'w') as nf_rf:
                        nf_rf.write(st_f)
                    plt.subplots(figsize=(16, 9), dpi=300)
                    plt.semilogy(np.arange(0, current_epoch, 1),
                                 loss_record_f[0, :current_epoch, 4], color="r", label="Training")
                    plt.semilogy(np.arange(1, current_epoch, 1),
                                 loss_record_f[1, 1:current_epoch, 4], color="b", label="Validation")
                    plt.xlabel("Training steps", fontsize=20)
                    plt.ylabel("Loss of magnetic moment", fontsize=20)
                    plt.legend(fontsize=18)
                    plt.grid(True, linestyle="--", linewidth=1)
                    plt.savefig(os.path.join(path_job, "_CHGNET_Loss_Magnetic moment.png"))
                    plt.close()


                if "**" in data_pf:
                    exit_loop = True
                    
        path_test = os.path.join(path_job, "test_result.json")
        if os.path.exists(path_test):
            exit_loop = True

        num_line_data = len(data_all) - 1
        time.sleep(interval)


def linear_reg(xf, yf):
    xf_avg, yf_avg = np.average(xf), np.average(yf)
    slope_f = np.sum((xf - xf_avg) * (yf - yf_avg)) / np.sum((xf - xf_avg) ** 2)
    inter_f = yf_avg - slope_f * xf_avg

    predict_yf = slope_f * xf + inter_f
    r_sq_f = 1 - np.sum((predict_yf - yf) ** 2) / np.sum((yf - yf_avg) ** 2)

    return slope_f, inter_f, r_sq_f


def export_testing(path_job, training_target):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.ioff()

    with open(os.path.join(path_job, "test_result.json"), 'r', encoding="utf-8") as nf:
        data_test = nf.read()

    data_test = json.loads(data_test)

    # Energy per atom
    energy_f = np.zeros((len(data_test), 2))
    st_f = "# " + st_str + "\n\n      From DFT     Prediction  (eV/atoms)\n"
    for nif in range(len(data_test)):
        data_f = data_test[nif]["energy"]
        energy_f[nif, :] = np.array([data_f["ground_truth"], data_f["prediction"]])
        st_f += (num_printer(energy_f[nif, 0], 14, 6, 0) +
                 num_printer(energy_f[nif, 1], 14, 6, 0) + "\n")
    with open(os.path.join(path_job, "__CHGNet_Energy_testing.log"), 'w') as nf_rf:
        nf_rf.write(st_f)
    linear_reg_v = linear_reg(energy_f[:, 0], energy_f[:, 1])
    plt.subplots(figsize=(12, 9), dpi=300)
    plt.scatter(energy_f[:, 0], energy_f[:, 1], color="r", alpha=0.7)
    plt.plot([np.min(energy_f), np.max(energy_f)], [np.min(energy_f), np.max(energy_f)],
             linewidth=2, linestyle="--", color="k")
    plt.xlabel("From DFT (eV/atoms)", fontsize=20)
    plt.ylabel("From MLP (eV/atoms)", fontsize=20)
    plt.title("Energy per atoms  ;  Number of data: " + num_printer(energy_f.shape[0], 8, 0, 0) +
              "\nSlope: " + num_printer(linear_reg_v[0], 6, 3, 0) +
              " ; Intercept: " + num_printer(linear_reg_v[1], 13, 3, 1) +
              " ; R_sq: " + num_printer(linear_reg_v[2], 6, 3, 0), fontsize=20)
    plt.grid(True, linestyle="--", linewidth=1, alpha=0.7)
    plt.savefig(os.path.join(path_job, "__CHGNet_Energy_testing.png"))
    plt.close()

    # Force
    force_f = np.zeros((0, 2))
    st_f = "# " + st_str + "\n\n      From DFT     Prediction  (eV/Angstrom)\n"
    for nif in range(len(data_test)):
        data_f = data_test[nif]["force"]
        force_new_f = np.concatenate((np.array([data_f["ground_truth"]]).reshape(-1)[:, np.newaxis],
                                      np.array([data_f["prediction"]]).reshape(-1)[:, np.newaxis]), axis=1)
        force_f = np.concatenate((force_f, force_new_f), axis=0)
        for nif2 in range(force_new_f.shape[0]):
            st_f += (num_printer(force_new_f[nif2, 0], 14, 6, 0) +
                     num_printer(force_new_f[nif2, 1], 14, 6, 0) + "\n")
    with open(os.path.join(path_job, "__CHGNet_Force_testing.log"), 'w') as nf_rf:
        nf_rf.write(st_f)
    linear_reg_v = linear_reg(force_f[:, 0], force_f[:, 1])
    plt.subplots(figsize=(12, 9), dpi=300)
    plt.scatter(force_f[:, 0], force_f[:, 1], color="b", alpha=0.7)
    plt.plot([np.min(force_f), np.max(force_f)], [np.min(force_f), np.max(force_f)],
             linewidth=2, linestyle="--", color="k")
    plt.xlabel("From DFT (eV/Angstrom)", fontsize=20)
    plt.ylabel("From MLP (eV/Angstrom)", fontsize=20)
    plt.title("Force  ;  Number of data: " + num_printer(force_f.shape[0], 8, 0, 0) +
              "\nSlope: " + num_printer(linear_reg_v[0], 6, 3, 0) +
              " ; Intercept: " + num_printer(linear_reg_v[1], 13, 3, 1) +
              " ; R_sq: " + num_printer(linear_reg_v[2], 6, 3, 0), fontsize=20)
    plt.grid(True, linestyle="--", linewidth=1, alpha=0.7)
    plt.savefig(os.path.join(path_job, "__CHGNet_Force_testing.png"))
    plt.close()

    # Stress
    stress_f = np.zeros((len(data_test), 2))
    if "s" in training_target:
        st_f = "# " + st_str + "\n\n      From DFT     Prediction  (GPa)\n"
        for nif in range(len(data_test)):
            stress_f[nif, :] = np.array([-np.average(np.diag(np.array(data_test[nif]["stress"]["ground_truth"]))),
                                         -np.average(np.diag(np.array(data_test[nif]["stress"]["prediction"])))])
            st_f += (num_printer(stress_f[nif, 0], 14, 6, 0) +
                     num_printer(stress_f[nif, 1], 14, 6, 0) + "\n")
        with open(os.path.join(path_job, "__CHGNet_Hydrostatic stress_testing.log"), 'w') as nf_rf:
            nf_rf.write(st_f)
        linear_reg_v = linear_reg(stress_f[:, 0], stress_f[:, 1])
        plt.subplots(figsize=(12, 9), dpi=300)
        plt.scatter(stress_f[:, 0], stress_f[:, 1], color="r", alpha=0.7)
        plt.plot([np.min(stress_f), np.max(stress_f)], [np.min(stress_f), np.max(stress_f)],
                 linewidth=2, linestyle="--", color="k")
        plt.xlabel("From DFT (GPa)", fontsize=20)
        plt.ylabel("From MLP (GPa)", fontsize=20)
        plt.title("Hydrostatic stress  ;  Number of data: " + num_printer(stress_f.shape[0], 8, 0, 0) +
                  "\nSlope: " + num_printer(linear_reg_v[0], 6, 3, 0) +
                  " ; Intercept: " + num_printer(linear_reg_v[1], 13, 3, 1) +
                  " ; R_sq: " + num_printer(linear_reg_v[2], 6, 3, 0), fontsize=20)
        plt.grid(True, linestyle="--", linewidth=1, alpha=0.7)
        plt.savefig(os.path.join(path_job, "__CHGNet_Hydrostatic stress_testing.png"))
        plt.close()

    # Magnetic moment
    magmom_f = np.zeros((0, 2))
    if "m" in training_target:
        st_f = "# " + st_str + "\n\n      From DFT     Prediction  (mu_B)\n"
        for nif in range(len(data_test)):
            data_f = data_test[nif]["mag"]
            magmom_new_f = np.concatenate((np.array(data_f["ground_truth"])[:, np.newaxis],
                                           np.array(data_f["prediction"])[:, np.newaxis]), axis=1)
            magmom_f = np.concatenate((magmom_f, magmom_new_f), axis=0)
            for nif2 in range(magmom_new_f.shape[0]):
                st_f += (num_printer(magmom_new_f[nif2, 0], 14, 6, 0) +
                         num_printer(magmom_new_f[nif2, 1], 14, 6, 0) + "\n")
        with open(os.path.join(path_job, "__CHGNet_Magmetic moment_testing.log"), 'w') as nf_rf:
            nf_rf.write(st_f)
        linear_reg_v = linear_reg(magmom_f[:, 0], magmom_f[:, 1])
        plt.subplots(figsize=(12, 9), dpi=300)
        plt.scatter(magmom_f[:, 0], magmom_f[:, 1], color="b", alpha=0.7)
        plt.plot([np.min(magmom_f), np.max(magmom_f)], [np.min(magmom_f), np.max(magmom_f)],
                 linewidth=2, linestyle="--", color="k")
        plt.xlabel("From DFT (mu_B)", fontsize=20)
        plt.ylabel("From MLP (mu_B)", fontsize=20)
        plt.title("Magmetic moment  ;  Number of data: " + num_printer(magmom_f.shape[0], 8, 0, 0) +
                  "\nSlope: " + num_printer(linear_reg_v[0], 6, 3, 0) +
                  " ; Intercept: " + num_printer(linear_reg_v[1], 13, 3, 1) +
                  " ; R_sq: " + num_printer(linear_reg_v[2], 6, 3, 0), fontsize=20)
        plt.grid(True, linestyle="--", linewidth=1, alpha=0.7)
        plt.savefig(os.path.join(path_job, "__CHGNet_Magmetic moment_testing.png"))
        plt.close()


# Start point
if __name__ == "__main__":
    st_time = time.time()
    st_str = ("Modify by Zhong-Lun Li Sep. 16 2025\n# Department of Chemical Engineering, "
              "National Taiwan University of Science and Technology, Taipei 106, Taiwan")
    Path_job = "."
    Path_log = os.path.join(Path_job, "CHGNET_training_results.log")
    Path_out = os.path.join(Path_job, "CHGNET_tmp.log")
    with open(Path_log, 'w', encoding="utf-8") as nf_r:
        nf_r.write("# " + st_str + "\n\n")

    RAT_TRA = 0.80
    RAT_VAL = 0.10
    LEA_R = 1E-2
    CRT = "MSE"
    LRS = "CosLR"
    EPOCH = 50
    FPTO = 100
    if len(sys.argv) > 1:
        BATCH = int(sys.argv[1])
    else:
        BATCH = 1
    RANDSEED = None

    if os.path.exists(os.path.join(Path_job, "Input_CHGNet_Training")):
        rd_cncar_t(os.path.join(Path_job, "Input_CHGNet_Training"))
    else:
        wr_cncar_t()

    Data_structures, Data_energies_per_atom, Data_forces, Data_stresses, Data_magmoms = [], [], [], [], []
    loss_data_s, loss_data_m = 0, 0
    for ni_path in glob.glob(f"{Path_job}/*.json"):  # for each .json file
        if os.path.basename(ni_path) == "test_result.json":
            continue
        dataset_dict = read_json(ni_path)

        for nii, nii_data in enumerate(dataset_dict["structure"]):
            Data_structures.append(Structure.from_dict(nii_data))

        Data_energies_per_atom += dataset_dict["energy_per_atom"]
        Data_forces += dataset_dict["force"]
        Data_stresses += dataset_dict["stress"]
        Data_magmoms += dataset_dict["magmom"]

        if len(dataset_dict["magmom"]) == 0:
            loss_data_m += 1
        if len(dataset_dict["stress"]) == 0:
            loss_data_s += 1

    if (loss_data_s != 0) & (loss_data_m != 0):
        Training_target = "ef"
        Dataset = StructureData(structures=Data_structures, energies=Data_energies_per_atom, forces=Data_forces)
        gen(Path_log, "Training targets: Energy and Force ...\n")
    elif (loss_data_s == 0) & (loss_data_m != 0):
        Training_target = "efs"
        Dataset = StructureData(structures=Data_structures, energies=Data_energies_per_atom,
                                forces=Data_forces, stresses=Data_stresses)
        gen(Path_log, "Training targets: Energy, Force and Stress ...\n")
    else:
        Training_target = "efsm"
        Dataset = StructureData(structures=Data_structures, energies=Data_energies_per_atom,
                                forces=Data_forces, stresses=Data_stresses, magmoms=Data_magmoms)
        gen(Path_log, "Training targets: Energy, Force, Stress and Magnetic moment ...\n")

    Train_loader, Val_loader, Test_loader = (
        get_train_val_test_loader(dataset=Dataset, batch_size=BATCH, train_ratio=RAT_TRA, val_ratio=RAT_VAL,
                                  num_workers=BATCH, rand_seed=RANDSEED))

    # Load pretrained CHGNet
    from chgnet.model import CHGNet
    Path_pot = os.path.join(Path_job, "Fine_Tune_Model.tar")
    if os.path.exists(Path_pot):
        checkpoint = torch.load(Path_pot, map_location="cpu")
        Model_CHGNET = CHGNet.from_dict(checkpoint["model"])
        gen(Path_log, 'Read the "Fine_Tune_Model.tar" archive '
                      'and continue training using the previously trained model...\n')
    else:
        Model_CHGNET = CHGNet.load()
        gen(Path_log, 'Unable to read "Fine_Tune_Model.tar" file, '
                      'starting training using pre-trained model...\n')

    Process_Training = Process(target=train_main,
                               args=(train_main_sub, Path_out,
                                     Path_job, Train_loader, Val_loader, Test_loader, Model_CHGNET,
                                     Training_target, LRS, CRT, EPOCH, LEA_R, FPTO))
    Process_Record = Process(target=train_record,
                             args=(Path_job, Path_log, Path_out, st_str, EPOCH, Training_target, 10))
    Process_Training.start()
    Process_Record.start()

    try:
        Process_Training.join()
        Process_Record.join()
    except KeyboardInterrupt:
        Process_Training.terminate()
        Process_Record.terminate()
        Process_Training.join()
        Process_Record.join()

    # plot the data from testing set
    export_testing(Path_job, Training_target)

    gen(Path_log, " ")
    gen(Path_log, "Training is over, program terminated! This took " +
        num_printer((time.time() - st_time) / 3600, 8, 2, 0) + " hours.")


