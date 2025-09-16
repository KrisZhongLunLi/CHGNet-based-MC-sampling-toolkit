# Modify by Zhong-Lun Li Sep. 16 2025,
# Department of Chemical Engineering, National Taiwan University of Science and Technology, Taipei 106, Taiwan

# The CHGNet model comes from this study:
# Deng, B., Zhong, P., Jun, K. et al. CHGNet as a pretrained universal neural network potential for
# charge-informed atomistic modelling. Nat Mach Intell 5, 1031–1041 (2023). https://doi.org/10.1038/s42256-023-00716-3

import numpy as np
import os, re, sys, json
from monty.io import zopen
from monty.os.path import zpath
from pymatgen.io.vasp.outputs import Oszicar, Vasprun


def parse_vasp_dir(
        base_dir: str,
        *,
        check_electronic_convergence: bool = True,
        save_path: str | None = None,
) -> dict[str, list]:
    """Parse VASP output files into structures and labels
    By default, the magnetization is read from mag_x from VASP,
    plz modify the code if magnetization is for (y) and (z).

    Args:
        base_dir (str): the directory of the VASP calculation outputs
        check_electronic_convergence (bool): if set to True, this function will raise
            Exception to VASP calculation that did not achieve electronic convergence.
            Default = True
        save_path (str): path to save the parsed VASP labels

    Raises:
        NotADirectoryError: if the base_dir is not a directory

    Returns:NotADirectoryError(f"{base_dir=} is not a directory"
        dict: a dictionary of lists with keys for structure, uncorrected_total_energy,
            energy_per_atom, force, magmom, stress.
    """
    if not os.path.isdir(base_dir):
        raise NotADirectoryError(f"{base_dir=} is not a directory")

    oszicar_path = zpath(f"{base_dir}/OSZICAR")
    vasprun_path = zpath(f"{base_dir}/vasprun.xml")
    outcar_path = zpath(f"{base_dir}/OUTCAR")
    if not os.path.exists(oszicar_path) or not os.path.exists(vasprun_path):
        raise RuntimeError(f"No data parsed from {base_dir}!")

    oszicar = Oszicar(oszicar_path)
    vasprun_orig = Vasprun(
        vasprun_path,
        parse_dos=False,
        parse_eigen=False,
        parse_projected_eigen=False,
        parse_potcar_file=False,
        exception_on_bad_xml=False,
    )

    charge, mag_x, mag_y, mag_z, header = [], [], [], [], []

    with zopen(outcar_path, encoding="utf-8", mode="rt") as file:
        all_lines = [line.strip() for line in file.readlines()]

    # For single atom systems, VASP doesn't print a total line, so
    # reverse parsing is very difficult
    # for SOC calculations only
    read_charge = read_mag_x = read_mag_y = read_mag_z = False
    mag_x_all = []
    ion_step_count = 0

    for clean in all_lines:
        if "magnetization (x)" in clean:
            ion_step_count += 1
        if read_charge or read_mag_x or read_mag_y or read_mag_z:
            if clean.startswith("# of ion"):
                header = re.split(r"\s{2,}", clean.strip())
                header.pop(0)
            elif re.match(r"\s*(\d+)\s+(([\d\.\-]+)\s+)+", clean):
                tokens = [float(token) for token in re.findall(r"[\d\.\-]+", clean)]
                tokens.pop(0)
                if read_charge:
                    charge.append(dict(zip(header, tokens, strict=True)))
                elif read_mag_x:
                    mag_x.append(dict(zip(header, tokens, strict=True)))
                elif read_mag_y:
                    mag_y.append(dict(zip(header, tokens, strict=True)))
                elif read_mag_z:
                    mag_z.append(dict(zip(header, tokens, strict=True)))
            elif clean.startswith("tot"):
                if ion_step_count == (len(mag_x_all) + 1):
                    mag_x_all.append(mag_x)
                read_charge = read_mag_x = read_mag_y = read_mag_z = False
        if clean == "total charge":
            read_charge = True
            read_mag_x = read_mag_y = read_mag_z = False
        elif clean == "magnetization (x)":
            mag_x = []
            read_mag_x = True
            read_charge = read_mag_y = read_mag_z = False
        elif clean == "magnetization (y)":
            mag_y = []
            read_mag_y = True
            read_charge = read_mag_x = read_mag_z = False
        elif clean == "magnetization (z)":
            mag_z = []
            read_mag_z = True
            read_charge = read_mag_x = read_mag_y = False
        elif re.search("electrostatic", clean):
            read_charge = read_mag_x = read_mag_y = read_mag_z = False

    if len(oszicar.ionic_steps) == len(mag_x_all):  # unfinished VASP job
        warnings.warn("Unfinished OUTCAR", stacklevel=2)
    elif len(oszicar.ionic_steps) == (len(mag_x_all) - 1):  # finished job
        mag_x_all.pop(-1)

    n_atoms = len(vasprun_orig.ionic_steps[0]["structure"])

    dataset = {
        "structure": [],
        "uncorrected_total_energy": [],
        "energy_per_atom": [],
        "force": [],
        "magmom": [],
        "stress": None if "stress" not in vasprun_orig.ionic_steps[0] else [],
    }

    sp_f = 1
    for index, ionic_step in enumerate(vasprun_orig.ionic_steps):
        if index < Start_frame:
            continue
        if End_frame < index:
            continue

        if (
                check_electronic_convergence
                and len(ionic_step["electronic_steps"]) >= vasprun_orig.parameters["NELM"]
        ):
            continue

        if sp_f < Interval_frame:
            sp_f += 1
            continue
        else:
            sp_f = 1

        dataset["structure"].append(ionic_step["structure"])
        dataset["uncorrected_total_energy"].append(ionic_step["e_0_energy"])
        dataset["energy_per_atom"].append(ionic_step["e_0_energy"] / n_atoms)
        dataset["force"].append(ionic_step["forces"])
        if mag_x_all != []:
            dataset["magmom"].append([site["tot"] for site in mag_x_all[index]])
        if "stress" in ionic_step:
            dataset["stress"].append(ionic_step["stress"])

    if dataset["uncorrected_total_energy"] == []:
        raise RuntimeError(f"No data parsed from {base_dir}!")

    if save_path is not None:
        save_dict = dataset.copy()
        save_dict["structure"] = [struct.as_dict() for struct in dataset["structure"]]
        write_json(save_dict, save_path)
    return dataset


def write_json(dct: dict, filepath: str) -> dict:
    """Write the JSON file.

    Args:
        dct (dict): dictionary to write
        filepath (str): file name of JSON to write.
    """

    def handler(obj: object) -> int | list | object:
        """Convert numpy int64 to int.

        Fixes TypeError: Object of type int64 is not JSON serializable
        reported in https://github.com/CederGroupHub/chgnet/issues/168.

        Returns:
            int | object: object for serialization
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except:
                return str(obj)

    with open(filepath, mode="w") as file:
        json.dump(dct, file, default=handler)


Path_job = input('Enter the folder path of the "VASP" calculation file: \n')
Path_save = input("Enter the path of the file to be exported: \n")
Start_frame = input("Enter the starting frame number (default: 0): \n")
if len(Start_frame) == 0:
    Start_frame = 0
else:
    Start_frame = int(Start_frame)
End_frame = input("Enter the ending frame number (default: last): \n")
if len(End_frame) == 0:
    End_frame = 1E12
else:
    End_frame = int(End_frame)
Interval_frame = input("Enter the sampling interval (default: 1): \n")
if len(Interval_frame) == 0:
    Interval_frame = 1
else:
    Interval_frame = int(Interval_frame)

N_file = sum(1 for f in os.listdir(Path_save) if f.endswith(".json")) + 1
Path_save_file = os.path.join(Path_save, "dataset_" + str(N_file) + ".json")
while os.path.exists(Path_save_file):
    N_file += 1
    Path_save_file = os.path.join(Path_save, "dataset_" + str(N_file) + ".json")
dataset_dict = parse_vasp_dir(base_dir=Path_job, save_path=Path_save_file)

