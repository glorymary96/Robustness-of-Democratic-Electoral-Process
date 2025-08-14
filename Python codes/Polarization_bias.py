#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:07:02 2023

@author: glory
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import numpy as np
import multiprocessing as mp
import timeit
import itertools
import shutil

start = timeit.default_timer()


# Generating natural opinions
# num_type = 0: Imposing bias with \mu
# num_type = 1: Imposing bias with p (change in weights)
# bi_mean: \Delta/2
# sd: standard deviation, always set to 0.2
def Natural_opinion(num_type):
    if num_type == 0:
        for nos in range(Num_simulation):
            y1 = np.random.normal(-bi_mean, sd, Nn)
            y2 = np.random.normal(bi_mean, sd, Np)
            x0 = np.concatenate((y1, y2))
            x0 = np.sort(x0)
            for mu_val_num in range(len(MU_VAL)):
                x = x0 + MU_VAL[mu_val_num]
                np.savez("DATA/NO_" + str(mu_val_num) + "_" + str(nos) + ".npz", X0=x)
    elif num_type == 1:
        for b in range(nbias):  # Natural opinions
            Nnn = Nn - 2 * b
            Npp = Np + 2 * b
            Array_pos.append(Npp)
            Array_neg.append(Nnn)
            for nos in range(Num_simulation):
                y1 = np.random.normal(-bi_mean, sd, Nnn)
                y2 = np.random.normal(bi_mean, sd, Npp)
                x0 = np.concatenate((y1, y2))
                x0 = np.sort(x0)
                np.savez("DATA/NO_" + str(b) + "_" + str(nos) + ".npz", X0=x0)


def outcome(h):
    d, nos, t = int(M[h][0]), int(M[h][1]), int(M[h][2])
    data = np.load("DATA/NO_" + str(d) + "_" + str(nos) + ".npz")
    x0 = data["X0"]
    e = e_range[t]
    D = np.tile(x0, (len(x0), 1))
    A = (abs(D - np.transpose(D)) < e) * 1 - np.identity(len(x0))
    Mat_inv = np.linalg.inv(
        np.matmul(
            np.diag(np.array([1 / i if i != 0 else 0 for i in np.sum(A, axis=1)])),
            np.diag(np.sum(A, axis=1)) - A,
        )
        + np.identity(len(x0))
    )
    Y = np.matmul(Mat_inv, x0)
    P_ini = np.sum(Y > 0)
    N_ini = N1 - P_ini

    if N_ini > P_ini:
        x0 = -x0
        P_ini, N_ini = N_ini, P_ini
        Y = -Y

    B = np.zeros((nsim))

    # Uncomment this to include the Random strategy
    """
 # Random Influence
    for t2 in range(nsim): 
        y=Y.copy()
        p=P_ini.copy()
        n=N_ini.copy()    
        iter=0
        f=1
        while f==1 and iter<10:          
            f=0
            z=np.random.permutation(N1) # Random selection
            b=0
            while p>n and b!=N1-1:
                y=y+val_w0*Mat_inv[:,z[b]]
                p=np.sum(y>0)
                n=N1-p
                b=b+1
                if b==N1-1: 
                    f=1
                    iter=iter+1
        B[t2]=(iter*len(x0))+b          
    """
    mu = np.mean(B)
    sigma = np.std(B)

    # Influence by minimum value
    z = np.argsort(
        x0 + 1e2 * (x0 < 0) * (-x0)
    )  # Indices of the agents near to the neutral opinion
    p = P_ini.copy()
    n = N_ini.copy()
    y = Y.copy()
    iter = 0
    f = 1

    while f == 1 and iter < 10:
        f = 0
        b = 0
        while p > n and b != N1 - 1:
            y = y + val_w0 * Mat_inv[:, z[b]]
            p = np.sum(y > 0)
            n = N1 - p
            b = b + 1
            if b == N1 - 1:
                f = 1
                iter = iter + 1
    Num_agents_inf = (iter * len(x0)) + b

    np.savez(
        "DATA/Res_" + str(h) + ".npz",
        pos=d,
        nos=nos,
        eps=t,
        Num_agents_inf=Num_agents_inf,
        P_ini=P_ini,
        N_ini=N_ini,
        mu=mu,
        sigma=sigma,
    )


def Polarization():
    num_type = 0
    os.mkdir("DATA")

    Natural_opinion(num_type)

    pool = mp.Pool(mp.cpu_count())
    pool.map(outcome, range(len(M)))
    pool.close()

    NUM_AGENTS_INF = np.zeros((len(MU_VAL), Num_simulation, neps))
    P_INI = np.zeros((len(MU_VAL), Num_simulation, neps))
    N_INI = np.zeros((len(MU_VAL), Num_simulation, neps))
    MU = np.zeros((len(MU_VAL), Num_simulation, neps))
    SIGMA = np.zeros((len(MU_VAL), Num_simulation, neps))

    for h in range(len(M)):
        data = np.load("DATA/Res_" + str(h) + ".npz")
        m = data["pos"]
        nos = data["nos"]
        eps = data["eps"]
        NUM_AGENTS_INF[m, nos, eps] = data["Num_agents_inf"]
        P_INI[m, nos, eps] = data["P_ini"]
        N_INI[m, nos, eps] = data["N_ini"]
        SIGMA[m, nos, eps] = data["sigma"]
        MU[m, nos, eps] = data["mu"]

    np.savez(
        "FILES/Min_Polarization_" + str(sd_num) + "_" + str(bi_mean_num) + ".npz",
        NOS=Num_simulation,
        SD=SD,
        MU_VAL=MU_VAL,
        BI_MEAN=BI_MEAN,
        e_range=e_range,
        NUM_AGENTS_INF=NUM_AGENTS_INF,
        P_INI=P_INI,
        N_INI=N_INI,
        MU=MU,
        SIGMA=SIGMA,
    )
    shutil.rmtree("DATA")


def Proportion():
    num_type = 1
    os.mkdir("DATA")
    Natural_opinion(num_type)
    pool = mp.Pool(mp.cpu_count())
    pool.map(outcome, range(len(M)))
    pool.close()

    NUM_AGENTS_INF = np.zeros((nbias, Num_simulation, neps))
    P_INI = np.zeros((nbias, Num_simulation, neps))
    N_INI = np.zeros((nbias, Num_simulation, neps))
    MU = np.zeros((nbias, Num_simulation, neps))
    SIGMA = np.zeros((nbias, Num_simulation, neps))

    for i in range(Num_simulation * neps * nbias):
        data = np.load("DATA/Res_" + str(i) + ".npz")
        d = data["pos"]
        g = data["nos"]
        t = data["eps"]
        NUM_AGENTS_INF[d, g, t] = data["Num_agents_inf"]
        P_INI[d, g, t] = data["P_ini"]
        N_INI[d, g, t] = data["N_ini"]
        MU[d, g, t] = data["mu"]
        SIGMA[d, g, t] = data["sigma"]

    np.savez(
        "FILES/Min_Diffprop_" + str(sd_num) + "_" + str(bi_mean_num) + ".npz",
        NOS=Num_simulation,
        SD=SD,
        BI_MEAN=BI_MEAN,
        Array_pos=Array_pos,
        Array_neg=Array_neg,
        e_range=e_range,
        NUM_AGENTS_INF=NUM_AGENTS_INF,
        P_INI=P_INI,
        N_INI=N_INI,
        MU=MU,
        SIGMA=SIGMA,
    )
    shutil.rmtree("DATA")


if __name__ == "__main__":
    Num_simulation = 500  # Number of natural opinion
    Np = 1001  # Number of people with positive opinion
    Nn = 1000  # Number of people with negative opinion
    N1 = Nn + Np  # Total number of people
    neps = 51
    Conn_per = np.arange(3, 7, 1) * 0.01
    e_range = np.linspace(0, 1.5, neps)
    e_num = np.arange(0, neps, 1)
    eigeval = 1
    val_w0 = -0.1

    nsim = 30

    SD = np.linspace(0.2, 0.4, 1)

    for sd_num in range(len(SD)):
        sd = SD[sd_num]

        os.mkdir("FILES")
        MU_VAL = np.linspace(0, 0.08, 1)
        BI_MEAN = np.linspace(0, 0.6, 121)

        M = [
            (i, j, k)
            for i, j, k in itertools.product(
                range(0, len(MU_VAL), 1), range(0, Num_simulation, 1), range(0, neps, 1)
            )
        ]
        for bi_mean_num in range(len(BI_MEAN)):
            bi_mean = BI_MEAN[bi_mean_num]
            Polarization()

        EFFORT = np.zeros((len(BI_MEAN), len(MU_VAL), Num_simulation, len(e_range)))
        NUM_AGENTS_INFLUENCED = np.zeros(
            (len(BI_MEAN), len(MU_VAL), Num_simulation, len(e_range))
        )
        EFFORT_RAND = np.zeros(
            (len(BI_MEAN), len(MU_VAL), Num_simulation, len(e_range))
        )

        for bi_mean_num in range(len(BI_MEAN)):
            data = np.load(
                "FILES/Min_Polarization_"
                + str(sd_num)
                + "_"
                + str(bi_mean_num)
                + ".npz"
            )
            EFFORT[bi_mean_num] = data["NUM_AGENTS_INF"] * abs(val_w0)
            NUM_AGENTS_INFLUENCED[bi_mean_num] = data["NUM_AGENTS_INF"]
            EFFORT_RAND[bi_mean_num] = data["MU"] * abs(val_w0)

        np.savez(
            "VAR_FILES/Polarization_delta.npz",
            Num_ppl=N1,
            e_range=e_range,
            strength_w0=abs(val_w0),
            SD=SD[0],
            DELTA_vals=2 * BI_MEAN,
            MEAN_vals=MU_VAL,
            EFFORT=EFFORT,
            NUM_AGENTS_INFLUENCED=NUM_AGENTS_INFLUENCED,
            EFFORT_RAND=EFFORT_RAND,
        )
        shutil.rmtree("FILES")

        os.mkdir("FILES")
        MU_VAL = np.linspace(0, 0.05, 51)
        BI_MEAN = np.linspace(0, 0.25, 2)
        M = [
            (i, j, k)
            for i, j, k in itertools.product(
                range(0, len(MU_VAL), 1), range(0, Num_simulation, 1), range(0, neps, 1)
            )
        ]
        for bi_mean_num in range(len(BI_MEAN)):
            bi_mean = BI_MEAN[bi_mean_num]
            Polarization()

        EFFORT = np.zeros((len(BI_MEAN), len(MU_VAL), Num_simulation, len(e_range)))
        NUM_AGENTS_INFLUENCED = np.zeros(
            (len(BI_MEAN), len(MU_VAL), Num_simulation, len(e_range))
        )
        EFFORT_RAND = np.zeros(
            (len(BI_MEAN), len(MU_VAL), Num_simulation, len(e_range))
        )

        for bi_mean_num in range(len(BI_MEAN)):
            data = np.load(
                "FILES/Min_Polarization_"
                + str(sd_num)
                + "_"
                + str(bi_mean_num)
                + ".npz"
            )
            EFFORT[bi_mean_num] = data["NUM_AGENTS_INF"] * abs(val_w0)
            NUM_AGENTS_INFLUENCED[bi_mean_num] = data["NUM_AGENTS_INF"]
            EFFORT_RAND[bi_mean_num] = data["MU"] * abs(val_w0)

        np.savez(
            "VAR_FILES/Mean_variation.npz",
            Num_ppl=N1,
            e_range=e_range,
            strength_w0=abs(val_w0),
            SD=SD[0],
            DELTA_vals=2 * BI_MEAN,
            MEAN_vals=MU_VAL,
            EFFORT=EFFORT,
            NUM_AGENTS_INFLUENCED=NUM_AGENTS_INFLUENCED,
            EFFORT_RAND=EFFORT_RAND,
        )
        shutil.rmtree("FILES")

        os.mkdir("FILES")
        nbias = 51
        BI_MEAN = np.linspace(0.25, 0.5, 1)
        MU_VAL = np.linspace(0, 0.08, 1)
        M = [
            (i, j, k)
            for i, j, k in itertools.product(
                range(0, nbias, 1), range(0, Num_simulation, 1), range(0, neps, 1)
            )
        ]
        for bi_mean_num in range(len(BI_MEAN)):
            Array_pos = []
            Array_neg = []
            bi_mean = BI_MEAN[bi_mean_num]
            Proportion()

        EFFORT = np.zeros((len(BI_MEAN), nbias, Num_simulation, len(e_range)))
        NUM_AGENTS_INFLUENCED = np.zeros(
            (len(BI_MEAN), nbias, Num_simulation, len(e_range))
        )
        EFFORT_RAND = np.zeros((len(BI_MEAN), nbias, Num_simulation, len(e_range)))

        for bi_mean_num in range(len(BI_MEAN)):
            data = np.load(
                "FILES/Min_Diffprop_" + str(sd_num) + "_" + str(bi_mean_num) + ".npz"
            )
            EFFORT[bi_mean_num] = data["NUM_AGENTS_INF"] * abs(val_w0)
            EFFORT_RAND[bi_mean_num] = data["MU"] * abs(val_w0)
            NUM_AGENTS_INFLUENCED[bi_mean_num] = data["NUM_AGENTS_INF"]
            Array_pos = data["Array_pos"]
            Array_neg = data["Array_neg"]

        np.savez(
            "VAR_FILES/Proportion_variation.npz",
            Num_ppl=N1,
            e_range=e_range,
            strength_w0=abs(val_w0),
            SD=SD[0],
            DELTA_vals=2 * BI_MEAN,
            MEAN_vals=MU_VAL,
            EFFORT=EFFORT,
            NUM_AGENTS_INFLUENCED=NUM_AGENTS_INFLUENCED,
            Array_pos=Array_pos,
            Array_neg=Array_neg,
            EFFORT_RAND=EFFORT_RAND,
        )
        shutil.rmtree("FILES")

stop = timeit.default_timer()
print("Time: ", (stop - start) / (60 * 60))
