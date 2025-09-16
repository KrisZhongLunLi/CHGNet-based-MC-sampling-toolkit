from scm.plams import *
import numpy as np
import os
import shutil
import time
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Modify by Zhong-Lun Li Dec. 29 2024,
# Department of Chemical Engineering, National Taiwan University of Science and Technology, Taipei 106, Taiwan

T_min = -200  # the minimum temperature, in degree Celsius
T_max = 300  # the maximum temperature, in degree Celsius
T_Gibbs_energy = 25   # the temperature which use to calculate Gibbs free energy, in degree Celsius
n_T = 251  # temperature sampling quantity
n_x = 51  # mole fraction sampling quantity
interpolation_up_lim = 1E-6
extrapolation_up_point = 5

R_cons = 8.31446261815324  # in J / mol / K


def interpolation(v_ind, v_dep, x_ind):  # Find numerical values using interpolation
    if np.isin(x_ind, v_ind):
        y_dep = v_dep[np.argmin(np.abs(x_ind - v_ind))]
    elif np.std(v_ind) <= interpolation_up_lim:
        y_dep = np.mean(v_dep)
    elif np.std(v_dep) <= interpolation_up_lim:
        y_dep = np.mean(v_dep)
    elif (np.min(v_ind) <= x_ind) & (x_ind < np.max(v_ind)):  # interpolation
        x_index_lower = np.argmin(np.abs(max((x for x in v_ind - x_ind if x <= 0)) + x_ind - v_ind))
        x_index_upper = np.argmin(np.abs(min((x for x in v_ind - x_ind if x > 0)) + x_ind - v_ind))
        y_dep = ((x_ind - v_ind[x_index_lower]) / (v_ind[x_index_upper] - v_ind[x_index_lower]) *
                 (v_dep[x_index_upper] - v_dep[x_index_lower]) + v_dep[x_index_lower])
    else:  # extrapolation
        n_p = int(min([np.shape(v_ind)[0], extrapolation_up_point]))

        if (x_ind - v_ind[0]) * (v_ind[0] - v_ind[-1]) == 0:
            pass
        elif (x_ind - v_ind[0]) * (v_ind[0] - v_ind[-1]) > 0:
            v_ind = v_ind[: n_p]
            v_dep = v_dep[: n_p]
        else:
            v_ind = v_ind[-n_p:]
            v_dep = v_dep[-n_p:]

        # linear, y = m_li * x + k_li
        m_li = ((np.dot(v_ind, v_dep) - n_p * np.mean(v_ind) * np.mean(v_dep)) /
                (np.dot(v_ind, v_ind) - n_p * np.mean(v_ind) ** 2))
        k_li = np.mean(v_dep) - m_li * np.mean(v_ind)
        r_li_sq = (1 - np.sum((m_li * v_ind + k_li - v_dep) ** 2) /
                   np.sum((np.mean(v_dep) - v_dep) ** 2))
        y_dep_pre_li = m_li * x_ind + k_li

        # exp, y = k_ex * exp(m_ex * x)
        if all(v_dep > 0):
            kk_ex = 0
        else:
            kk_ex = np.min(v_dep) - interpolation_up_lim
        m_ex = ((np.dot(v_ind, np.log(v_dep - kk_ex)) - n_p * np.mean(v_ind) * np.mean(np.log(v_dep - kk_ex))) /
                (np.dot(v_ind, v_ind) - n_p * np.mean(v_ind) ** 2))
        k_ex = np.mean(np.log(v_dep - kk_ex)) - m_ex * np.mean(v_ind)
        r_ex_sq = (1 - np.sum((m_ex * v_ind + k_ex - np.log(v_dep - kk_ex)) ** 2) /
                   np.sum((np.mean(np.log(v_dep - kk_ex)) - np.log(v_dep - kk_ex)) ** 2))
        y_dep_pre_ex = np.exp(k_ex + m_ex * x_ind) + kk_ex

        # log, y = m_ln * ln(x) + k_ln
        if all(v_ind > 0) and x_ind > 0:
            kk_ln = 0
        else:
            kk_ln = min([np.min(v_ind), x_ind]) - interpolation_up_lim
        m_ln = ((np.dot(np.log(v_ind - kk_ln), v_dep) - n_p * np.mean(np.log(v_ind - kk_ln)) * np.mean(v_dep)) /
                (np.dot(np.log(v_ind - kk_ln), np.log(v_ind - kk_ln)) - n_p * np.mean(np.log(v_ind - kk_ln)) ** 2))
        k_ln = np.mean(v_dep) - m_ln * np.mean(np.log(v_ind - kk_ln))
        r_ln_sq = (1 - np.sum((m_ln * np.log(v_ind - kk_ln) + k_ln - v_dep) ** 2) /
                   np.sum((np.mean(v_dep) - v_dep) ** 2))
        y_dep_pre_ln = m_ln * np.log(x_ind - kk_ln) + k_ln

        if r_ex_sq <= r_ln_sq:
            r_ex_sq = 0
        else:
            r_ln_sq = 0
        y_dep = ((y_dep_pre_li * r_li_sq + y_dep_pre_ex * r_ex_sq + y_dep_pre_ln * r_ln_sq) /
                 (r_li_sq + r_ex_sq + r_ln_sq))

    return y_dep


def integral(x_v, y_v, dx, x_d, x_u):
    x_d = np.max(np.array([x_d, x_v[0]]))
    x_u = np.min(np.array([x_u, x_v[-1]]))

    int_v = 0
    x_dv_min, x_dv_max = np.shape(np.where(x_d > x_v))[1], np.shape(np.where(x_u >= x_v))[1]
    if np.shape(np.where(x_d == x_v))[1] == 0:
        int_v += (interpolation(x_v, y_v, x_d) + y_v[x_dv_min]) * (x_v[x_dv_min] - x_d) / 2
    if np.shape(np.where(x_u == x_v))[1] == 0:
        int_v += (y_v[x_dv_max - 1] + interpolation(x_v, y_v, x_u)) * (x_u - x_v[x_dv_max - 1]) / 2

    int_v += \
        (np.sum(y_v[x_dv_min: x_dv_max]) - (y_v[x_dv_min] + y_v[x_dv_max - 1]) / 2) * dx

    return int_v


def cosmo_pure(nif):
    comp_path = Name_coskf[nif]
    import sys
    sys.stdout = open(os.devnull, 'w')

    kf = KFFile(comp_path)
    f_melt_temp = np.array([kf.read("Compound Data", "Melting Point")])  # in K
    f_fusion = np.array([kf.read("Compound Data", "Hfusion")]) * 4184  # in J / mol

    settings = Settings()
    settings.input.property._h = 'PUREVAPORPRESSURE'
    compounds = [Settings()]
    compounds[0]._h = comp_path

    settings.input.temperature = (format(T_min + 273.15, '.3f') + " " +
                                  format(T_max + 273.15, '.3f') + " " +
                                  format(n_T - 1, '.0f'))
    # specify the compounds as the compounds to be used in the calculation
    settings.input.compound = compounds
    # create a job that can be run by COSMO-RS
    my_job = CRSJob(settings=settings)
    # run the job
    init()
    out = my_job.run()
    finish()

    res = out.get_results()
    f_vaporization = np.array(res["Enthalpy of vaporization"])[0, :] * 4184  # in J / mol
    f_vap_div_temp_sq = f_vaporization / Tp_abs ** 2  # in J / mol / K ** 2
    vp = np.zeros(n_T)
    for nif2 in range(n_T):
        tp = Tp_abs[nif2]
        if tp < Pure_boilT[nif]:
            vp[nif2] = -integral(Tp_abs, f_vap_div_temp_sq, (T_max - T_min) / (n_T - 1), tp, Pure_boilT[nif])
        elif tp > Pure_boilT[nif]:
            vp[nif2] = integral(Tp_abs, f_vap_div_temp_sq, (T_max - T_min) / (n_T - 1), Pure_boilT[nif], tp)
        else:
            vp[nif2] = 0

    vap_pressure = Pressure_set * np.exp(vp / R_cons)

    if os.path.exists(os.path.dirname(my_job.path)):
        shutil.rmtree(os.path.dirname(my_job.path))

    sys.stdout = sys.__stdout__

    return f_melt_temp, f_fusion, vap_pressure


def cosmo_mixture(c1, c2, c3, c4, pure_vapor_pressure):
    import sys

    concentration, gamma = np.zeros((4, n_x, n_x, n_x)), np.zeros((4, n_T, n_x, n_x, n_x))
    excess_enthalpy, excess_free_energy = np.zeros((n_T, n_x, n_x, n_x)), np.zeros((n_T, n_x, n_x, n_x))
    cp = 0
    for nif in range(n_x):
        sys.stdout = open(os.devnull, 'w')

        for nif2 in range(n_x - nif):
            for nif3 in range(n_x - nif - nif2):
                settings = Settings()
                settings.input.property._h = 'VAPORPRESSURE'

                # set the number of compounds
                compounds = [Settings() for i in range(4)]
                compounds[0]._h = Name_coskf[c1]
                compounds[1]._h = Name_coskf[c2]
                compounds[2]._h = Name_coskf[c3]
                compounds[3]._h = Name_coskf[c4]

                # set compound mole fractions
                settings.input.temperature = (format(T_min + 273.15, '.3f') + " " +
                                              format(T_max + 273.15, '.3f') + " " +
                                              format(n_T - 1, '.0f'))

                compounds[0].frac1 = xp[nif]
                compounds[1].frac1 = xp[nif2]
                compounds[2].frac1 = xp[nif3]
                compounds[3].frac1 = 1 - xp[nif] - xp[nif2] - xp[nif3]

                settings.input.compound = compounds
                my_job = CRSJob(settings=settings)
                init()
                out = my_job.run()
                finish()

                res = out.get_results()

                if os.path.exists(os.path.dirname(my_job.path)):
                    shutil.rmtree(os.path.dirname(my_job.path))

                concentration[:, nif, nif2, nif3] = (
                    np.array([xp[nif], xp[nif2], xp[nif3], 1 - xp[nif] - xp[nif2] - xp[nif3]]))
                gamma[:, :, nif, nif2, nif3] = np.array(res["gamma"])
                excess_enthalpy[:, nif, nif2, nif3] = np.array(res["excess H"])  # in kcal / mol
                excess_free_energy[:, nif, nif2, nif3] = np.array(res["excess G"])  # in kcal / mol

                del res

        sys.stdout = sys.__stdout__

        if cp <= nif + 1:
            print("The current COSMO calculation progress: " + format(cp / n_x * 100, '.0f') + " %")
            cp += n_x / min(50, n_x - 1)

    vap_pre = (pure_vapor_pressure[:, :, np.newaxis, np.newaxis, np.newaxis] *
               concentration[:, np.newaxis, :, :, :] * gamma)  # in bar
    excess_enthalpy *= 4184  # in J / mol
    excess_free_energy *= 4184  # in J / mol

    return concentration, gamma, vap_pre, excess_enthalpy, excess_free_energy


def num_printer(data, sp, dg, tp):
    # tp: full number: 0, scientific notation: 1
    if abs(data) <= 10 ** -(dg + 3):
        data = 0
    blank_str = ""
    if data > 0:  # positive
        if tp == 1:
            if data <= 1E-100:
                for nif in range(int(sp - dg - 7)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'E')
            elif data >= 1E+100:
                for nif in range(int(sp - dg - 7)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'E')
            else:
                for nif in range(int(sp - dg - 6)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'E')
        else:
            od = max(1, np.floor(np.log10(data)) + 1)
            if od > sp:
                return format(data, '.' + str(dg) + 'f')
            else:
                for nif in range(int(sp - od - dg - 1)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'f')
    elif data < 0:  # negative
        if tp == 1:
            if data >= -1E-100:
                for nif in range(int(sp - dg - 8)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'E')
            elif data <= -1E+100:
                for nif in range(int(sp - dg - 8)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'E')
            else:
                for nif in range(int(sp - dg - 7)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'E')
        else:
            od = max(1, np.floor(np.log10(-data)) + 1)
            if od > sp:
                return format(data, '.' + str(dg) + 'f')
            else:
                for nif in range(int(sp - od - dg - 2)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'f')
    else:  # 0
        if tp == 1:
            for nif in range(int(sp - dg - 6)):
                blank_str += " "
            return blank_str + format(data, '.' + str(dg) + 'E')
        else:
            if 1 > sp:
                return format(data, '.' + str(dg) + 'f')
            else:
                for nif in range(int(sp - dg - 2)):
                    blank_str += " "
                return blank_str + format(data, '.' + str(dg) + 'f')


def pt_plot(xpf, ypf, zpf, zpf_low, zpf_high, n_con, od, cmap, tl, lbl, comp_name):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'

    plt.plot([0, 1], [0, 0], color='k', linewidth=2)
    plt.plot([0, 0.5], [0, np.sqrt(3) / 2], color='k', linewidth=2)
    plt.plot([0.5, 1], [np.sqrt(3) / 2, 0], color='k', linewidth=2)
    tick_len = 0.03
    for nif in range(1, 5):
        p_tick = nif / 5
        plt.plot([p_tick, tick_len * 1 / 2 + p_tick],
                 [0, tick_len * np.sqrt(3) / 2], color='k', linewidth=1.5)
        plt.plot([1 - 1 / 2 * p_tick, 1 - 1 / 2 * p_tick - tick_len],
                 [p_tick * np.sqrt(3) / 2, p_tick * np.sqrt(3) / 2], color='k', linewidth=1.5)
        plt.plot([1 / 2 * (1 - p_tick), 1 / 2 * (1 - p_tick + tick_len)],
                 [np.sqrt(3) / 2 * (1 - p_tick), np.sqrt(3) / 2 * (1 - p_tick - tick_len)], color='k', linewidth=1.5)

        plt.text(-0.01 + p_tick, -0.03, format(p_tick, '.1f'), fontsize=10,
                 fontweight='bold', fontfamily='Arial', ha="center", va="center", color="black")
        plt.text(1.04 - p_tick / 2, np.sqrt(3) / 2 * p_tick, format(p_tick, '.1f'), fontsize=10,
                 fontweight='bold', fontfamily='Arial', ha="center", va="center", color="black")
        plt.text(0.47 - p_tick / 2, 0.89 - np.sqrt(3) / 2 * p_tick, format(p_tick, '.1f'), fontsize=10,
                 fontweight='bold', fontfamily='Arial', ha="center", va="center", color="black")

    pf = plt.contourf(xpf, ypf, zpf,
                     levels=np.linspace(zpf_low, zpf_high, 201), cmap=cmap)
    cbar = plt.colorbar(pf)
    ticks = cbar.get_ticks()
    if od == 0:
        formatted_ticks = [f"{tick:.0f}" for tick in ticks]
    else:
        formatted_ticks = [f"{tick:.1f}" for tick in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(formatted_ticks)
    cbar.set_label(tl, fontsize=12, fontweight='bold', fontfamily='Arial')

    pf = plt.contour(xpf, ypf, zpf, levels=np.linspace(zpf_low, zpf_high, n_con),
                     colors=[(0.3, 0.3, 0.3, 1)], linewidths=1, linestyles='dashed')
    if od == 0:
        pfc = plt.clabel(pf, inline=True, fontsize=10, fmt=lambda x: f"{x:.0f}")
    else:
        pfc = plt.clabel(pf, inline=True, fontsize=10, fmt=lambda x: f"{x:.1f}")
    for label in pfc:
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')
        label.set_color("black")

    plt.text(0.05, 0.85, lbl, fontsize=16,
             fontweight='bold', fontfamily='Arial', ha="center", va="center", color="black")
    plt.text(0.50, -0.08, comp_name[0], fontsize=14,
             fontweight='bold', fontfamily='Arial', ha="center", va="center", color="black")
    plt.text(0.80, 0.45, comp_name[1], fontsize=14,
             fontweight='bold', fontfamily='Arial', ha="left", va="center", color="black")
    plt.text(0.20, 0.45, comp_name[2], fontsize=14,
             fontweight='bold', fontfamily='Arial', ha="right", va="center", color="black")

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')


def main(c1, c2, c3, c4):
    # Calculate the properties of pure substance
    Pure_meltT, Pure_fusH, Pure_vap_pressure = np.zeros(4), np.zeros(4), np.zeros((4, n_T))
    Pure_meltT[0], Pure_fusH[0], Pure_vap_pressure[0, :] = cosmo_pure(c1)
    Pure_meltT[1], Pure_fusH[1], Pure_vap_pressure[1, :] = cosmo_pure(c2)
    Pure_meltT[2], Pure_fusH[2], Pure_vap_pressure[2, :] = cosmo_pure(c3)
    Pure_meltT[3], Pure_fusH[3], Pure_vap_pressure[3, :] = cosmo_pure(c4)


    # Calculate the properties of mixture
    Concentration, Gamma_mat, Pressure_mat, Excess_H_mat, Excess_G_mat = (
        cosmo_mixture(c1, c2, c3, c4, Pure_vap_pressure))
    # Concentration: [4, 1st comp., 2nd comp., 3rd comp.]
    # Gamma_mat, Pressure_mat: [4, temp., 1st comp., 2nd comp., 3rd comp.]
    # Excess_H_mat, Excess_G_mat: [temp., 1st comp., 2nd comp., 3rd comp.]

    # Free energy calculation
    G_mix = np.sum(Concentration * np.log(np.where(Concentration <= 0, np.array([[[[1]]]]), Concentration)), axis=0)
    G_mix *= R_cons * (T_Gibbs_energy + 273.15)  # in J / mol
    G_mix_plot = np.zeros(int(n_x * (1 + n_x) * (2 + n_x) / 6))
    gn = 0
    for ni in range(n_x):
        for nii in range(n_x - ni):
            for ni3 in range(n_x - ni - nii):
                G_mix[ni, nii, ni3] += interpolation(Tp, Excess_G_mat[:, ni, nii, ni3], T_Gibbs_energy)  # in J / mol
                G_mix_plot[int(gn)] = G_mix[ni, nii, ni3]
                gn += 1


    # Calculate the solubility
    H_sol = Pure_fusH[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]  # in J / mol
    # H_sol: the enthalpy of solvation from solid to solvation, [solute type, temperature, mole fraction]
    activity_mat = Gamma_mat * Concentration[:, np.newaxis, :, :, :]
    T_sat = (Pure_meltT[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] ** -1 -
             np.log(np.where(activity_mat <= 0, np.array([[[[[1]]]]]), activity_mat)) * R_cons / H_sol) ** -1
    T_sat = np.where(activity_mat == 0, np.array([[[[[0]]]]]), T_sat)
    T_sat_max = np.max(T_sat, axis=0)
    T_sat_final = np.zeros((n_x, n_x, n_x))
    xy_ternary = np.zeros((2, int(n_x * (1 + n_x) * (2 + n_x) / 6), 3))
    # xy_ternary: mole fraction, coordinate of liquid, coordinate of vapor
    T_sat_final_plot = np.zeros(int(n_x * (1 + n_x) * (2 + n_x) / 6))
    gn = 0
    for ni in range(n_x):
        for nii in range(n_x - ni):
            for ni3 in range(n_x - ni - nii):
                T_sat_final[ni, nii, ni3] = interpolation(T_sat_max[:, ni, nii, ni3] - Tp_abs, Tp, 0) + 273.15
                xy_ternary[0, int(gn), :] = np.array([xp[ni], xp[nii], xp[ni3]])
                T_sat_final_plot[int(gn)] = T_sat_final[ni, nii, ni3]
                gn += 1


    Pressure_total_mat = np.sum(Pressure_mat, axis=0)
    y_mat_hold = Pressure_mat / np.where(Pressure_total_mat == 0, np.array([[[[1]]]]), Pressure_total_mat)

    Boiling_mat = np.zeros((n_x, n_x, n_x))
    Boiling_mat_plot = np.zeros(int(n_x * (1 + n_x) * (2 + n_x) / 6))
    y_mat = np.zeros((4, n_x, n_x, n_x))  # [mole fraction of vapor, each position]
    gn = 0
    for ni in range(n_x):
        for nii in range(n_x - ni):
            for ni3 in range(n_x - ni - nii):
                Boiling_mat[ni, nii, ni3] = interpolation(Pressure_total_mat[:, ni, nii, ni3], Tp_abs, Pressure_set)
                Boiling_mat_plot[int(gn)] = Boiling_mat[ni, nii, ni3]
                gn += 1

                y_mat_n = np.zeros(4)
                for ni4 in range(4):
                    y_mat_n[ni4] = interpolation(Pressure_total_mat[:, ni, nii, ni3],
                                                 y_mat_hold[ni4, :, ni, nii, ni3], Pressure_set)
                y_mat_n = np.where(y_mat_n < 0, np.array([0]), y_mat_n)
                y_mat_n = np.where(y_mat_n > 1, np.array([1]), y_mat_n)
                y_mat_n /= np.where(np.sum(y_mat_n, axis=0) == 0, np.array([1]), np.sum(y_mat_n, axis=0))
                y_mat[:, ni, nii, ni3] = y_mat_n


    # write the data
    st_array_x, st_array_mt = np.zeros((0, 3)), np.zeros(0)
    st = ("                         Mole fraction of                          Gibbs free energy of mixing    "
          "Melting temperature    Boiling temperature \n"
          "   x1      x2      x3      x4       y1      y2      y3      y4"
          "               (kJ/mol)                     (K)                    (K)")
    for ni in range(n_x):
        for nii in range(n_x - ni):
            for ni3 in range(n_x - ni - nii):
                st_array_x = np.concatenate((st_array_x, np.array([[xp[ni], xp[nii], xp[ni3]]])), axis=0)
                st_array_mt = np.concatenate((st_array_mt, np.array([T_sat_final[ni, nii, ni3]])))
                st += ("\n" + num_printer(xp[ni], 6, 3, 0) +
                       num_printer(xp[nii], 8, 3, 0) +
                       num_printer(xp[ni3], 8, 3, 0) +
                       num_printer(1 - xp[ni]  - xp[nii] - xp[ni3], 8, 3, 0) +
                       num_printer(y_mat[0, ni, nii, ni3], 9, 3, 0) +
                       num_printer(y_mat[1, ni, nii, ni3], 8, 3, 0) +
                       num_printer(y_mat[2, ni, nii, ni3], 8, 3, 0) +
                       num_printer(1 - np.sum(y_mat[:, ni, nii, ni3]), 8, 3, 0) +
                       num_printer(G_mix[ni, nii, ni3] / 1000, 23, 6, 0) +
                       num_printer(T_sat_final[ni, nii, ni3], 26, 6, 0) +
                       num_printer(Boiling_mat[ni, nii, ni3], 23, 6, 0))

    m_x1 = st_array_x[np.array(np.where(st_array_mt == np.min(st_array_mt))), 0][0]
    m_x2 = st_array_x[np.array(np.where(st_array_mt == np.min(st_array_mt))), 1][0]
    m_x3 = st_array_x[np.array(np.where(st_array_mt == np.min(st_array_mt))), 2][0]
    m_x4 = 1 - m_x1 - m_x2 - m_x3
    xp_p = np.array(np.where(np.min(st_array_mt) == T_sat_final))

    path_results = os.path.join(path, "COSMO_Quaternary_Results")
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    ns_1 = os.path.splitext(os.path.basename(Name_coskf[c1]))[0]
    ns_2 = os.path.splitext(os.path.basename(Name_coskf[c2]))[0]
    ns_3 = os.path.splitext(os.path.basename(Name_coskf[c3]))[0]
    ns_4 = os.path.splitext(os.path.basename(Name_coskf[c4]))[0]
    name_system = ns_1 + "-" + ns_2 + "-" + ns_3 + "-" + ns_4
    with open(os.path.join(path_results, name_system + ".log"), 'w') as wr:
        wr.write(st)

    # plot
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1 - m_x1, 100),
                                 np.linspace(0, np.sqrt(3) / 2 * (1 - m_x1), 100))

    xy_ternary_n = np.zeros((2, np.sum(xy_ternary[0, :, 0] == m_x1), 2))
    xy_ternary_n[0, :, :] = xy_ternary[0, xy_ternary[0, :, 0] == m_x1, 1:]
    xy_ternary_n[1, :, :] = np.dot(xy_ternary_n[0, :, :],
                                   np.array([[1, 0], [1 / 2, np.sqrt(3) / 2]])) * float(1 - m_x1)
    G_mix_mat_plot_grid = griddata((xy_ternary_n[1, :, 0], xy_ternary_n[1, :, 1]),
                                   G_mix_plot[xy_ternary[0, :, 0] == m_x1], (grid_x, grid_y),
                                   method='cubic')  # in J / mol

    Eutectic_mat_plot_grid = griddata((xy_ternary_n[1, :, 0], xy_ternary_n[1, :, 1]),
                                     T_sat_final_plot[xy_ternary[0, :, 0] == m_x1], (grid_x, grid_y),
                                      method='cubic') - 273.15
    Bubble_mat_plot_grid = griddata((xy_ternary_n[1, :, 0], xy_ternary_n[1, :, 1]),
                                    Boiling_mat_plot[xy_ternary[0, :, 0] == m_x1], (grid_x, grid_y),
                                    method='cubic') - 273.15

    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(7, 3, figure=fig)
    plt.subplot(gs[0: 2, :])
    plt.text(0.5, 0.90, name_system + " Quaternary system",
             fontsize=24, fontweight='bold', fontfamily='Arial', ha="center", va="center", color="black")

    plt.text(0.05, 0.65, ns_1 + "  :  " + ns_2 + "  :  " + ns_3 + "  :  " + ns_4 +
             "  =  " + format(m_x1[0], '.2f') + " : " + format(m_x2[0], '.2f') +
             " : " + format(m_x3[0], '.2f') + " : " + format(m_x4[0], '.2f'),
             fontsize=12, fontweight='bold', fontfamily='Arial', ha="left", va="center", color="black")
    plt.text(0.05, 0.35, "The eutectic temperature: " +
             format(np.min(st_array_mt) - 273.15, '.0f') + " °C",
             fontsize=12, fontweight='bold', fontfamily='Arial', ha="left", va="center", color="black")
    plt.text(0.05, 0.15, "The bubble temperature: " +
             format(Boiling_mat[xp_p[0][0], xp_p[1][0], xp_p[2][0]] - 273.15, '.0f') + " °C",
             fontsize=12, fontweight='bold', fontfamily='Arial', ha="left", va="center", color="black")
    plt.text(0.50, 0.25, "The liquid window of the eutectic point: " +
             format(Boiling_mat[xp_p[0][0], xp_p[1][0], xp_p[2][0]] - np.min(st_array_mt), '.0f') + " °C",
             fontsize=12, fontweight='bold', fontfamily='Arial', ha="center", va="center", color="black")
    plt.text(5 / 6, 0.25, "Gibbs energy of mixture: " +
             format(G_mix[xp_p[0][0], xp_p[1][0], xp_p[2][0]] / 1000, '.2f') + " kJ / mol",
             fontsize=12, fontweight='bold', fontfamily='Arial', ha="center", va="center", color="black")

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    plt.subplot(gs[2:, 0])
    pt_plot(grid_x, grid_y, Eutectic_mat_plot_grid,
            np.min(T_sat_final_plot[xy_ternary[0, :, 0] == m_x1]) - 273.15,
            np.max(T_sat_final_plot[xy_ternary[0, :, 0] == m_x1]) - 273.15, 10,
            0, "Blues_r", "Temperature (°C)", "(a)", np.array([ns_2, ns_3, ns_4]))

    plt.subplot(gs[2:, 1])
    pt_plot(grid_x, grid_y, Bubble_mat_plot_grid,
            np.min(Boiling_mat_plot[xy_ternary[0, :, 0] == m_x1]) - 273.15,
            np.max(Boiling_mat_plot[xy_ternary[0, :, 0] == m_x1]) - 273.15, 10,
            0, "Reds", "Temperature (°C)", "(b)", np.array([ns_2, ns_3, ns_4]))

    plt.subplot(gs[2:, 2])
    pt_plot(grid_x, grid_y, G_mix_mat_plot_grid / 1000, np.min(G_mix_plot[xy_ternary[0, :, 0] == m_x1]) / 1000,
            np.max(G_mix_plot[xy_ternary[0, :, 0] == m_x1]) / 1000, 5,
            1, "coolwarm_r", "Gibbs energy of mixture (kJ / mol)", "(c)", np.array([ns_2, ns_3, ns_4]))

    plt.tight_layout()

    plt.savefig(os.path.join(path_results, name_system + ".png"), dpi=600, bbox_inches='tight')

    plt.close()


# Start point
database_path = os.path.join(os.environ["SCM_PKG_ADFCRSDIR"], "ADFCRS-2018")
if not os.path.exists(database_path):
    raise OSError(f"The provided path does not exist. Exiting.")

# Read the file path
path = input('Please enter the path to the ".coskf" file folder\n')
if not os.path.isdir(path):
    exit(0)

Name_coskf = [os.path.join(path, "PC.coskf"), os.path.join(path, "DEC.coskf"), os.path.join(path, "MA.coskf"),
              os.path.join(path, "EA.coskf"), os.path.join(path, "MP.coskf"), os.path.join(path, "GBL.coskf"),
              os.path.join(path, "DOL.coskf"), os.path.join(path, "THF.coskf"), os.path.join(path, "TMS.coskf"),
              os.path.join(path, "TTE.coskf")]
Pure_boilT = [514.8, 400.2, 329.5, 350.3, 353.0, 477.0, 347.7, 338.2, 558.2, 366.4]  # in Kelvin
Pressure_set = 1.01325  # in bar

xp = np.linspace(0, 1, n_x)
Tp = np.linspace(T_min, T_max, n_T)
Tp_abs = Tp + 273.15

if len(Name_coskf) != len(Pure_boilT):
    exit(0)

if __name__ == "__main__":
    start_time = time.time()
    for nmi1 in range(len(Name_coskf)):
        for nmi2 in range(nmi1 + 1, len(Name_coskf)):
            if (nmi1 == 0) or (nmi1 == 8):
                continue
            if (nmi2 == 0) or (nmi2 == 8):
                continue
            st_t = time.time()
            main(nmi1, nmi2, 0, 8)
            Name_system = ((os.path.splitext(os.path.basename(Name_coskf[nmi1]))[0]) + "-" +
                           (os.path.splitext(os.path.basename(Name_coskf[nmi2]))[0]) + "-" +
                           os.path.splitext(os.path.basename(Name_coskf[0]))[0] + "-" +
                           os.path.splitext(os.path.basename(Name_coskf[8]))[0])
            print("Calculated: " + Name_system + ", time: " + format((time.time() - st_t) / 60, '.2f') + " mins\n")

    spend_time = time.time() - start_time
    spend_time_hr = np.floor(spend_time / 3600).astype(int)
    spend_time -= 3600 * spend_time_hr
    spend_time_min = np.floor(spend_time / 60).astype(int)
    spend_time -= 60 * spend_time_min
    print("Process: " + format(spend_time_hr, '.0f') + " hr " +
          format(spend_time_min, '.0f') + " min " + format(spend_time, '.1f') + " sec")

