#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:56:14 2022

@author: glory
"""

# Validation of the model

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
import pandas as pd

start = timeit.default_timer()


# Function to generate the natural opinion changing the weights of both gaussian distributions
def Nat_type1(R):
    X0 = []
    for j in range(len(R)):
        r1 = int(np.round(R[j] * 0.01 * NP_district))
        r2 = int(NP_district - r1)
        y1 = np.random.normal(-bi_mean, sd, r1)
        y2 = np.random.normal(bi_mean, sd, r2)
        x = np.concatenate((y1, y2))
        x = np.sort(x)
        d = max(x[1 : len(x)] - x[0 : len(x) - 1])
        while d > e_range[0]:
            # print('Entered',d)
            y1 = np.random.normal(-bi_mean, sd, r1)
            y2 = np.random.normal(bi_mean, sd, r2)
            x = np.concatenate((y1, y2))
            x = np.sort(x)
            d = max(x[1 : len(x)] - x[0 : len(x) - 1])
        X0.append(x)
    return X0


# Function to generate the natural opinion by shifting the mean of a biguassian distribution
def Nat_type2(R):
    nop1 = int(np.ceil(NP_district / 2))
    nop2 = NP_district - nop1
    X0 = []
    for j in range(len(R)):
        r1 = int(np.round(R[j] * 0.01 * NP_district))
        y1 = np.random.normal(-bi_mean, sd, nop1)
        y2 = np.random.normal(bi_mean, sd, nop2)
        x = np.concatenate((y1, y2))
        x = np.sort(x)
        d = max(x[1 : len(x)] - x[0 : len(x) - 1])
        while d > e_range[0]:
            y1 = np.random.normal(-bi_mean, sd, nop1)
            y2 = np.random.normal(bi_mean, sd, nop2)
            x = np.concatenate((y1, y2))
            x = np.sort(x)
            d = max(x[1 : len(x)] - x[0 : len(x) - 1])
        if r1 == 0:
            delta = -x[r1]
        elif r1 == NP_district:
            delta = -x[r1 - 1]
        else:
            delta = -(x[r1 - 1] + x[r1]) / 2
        x = x + delta
        X0.append(x)
    return X0


# Function to generate the natural opinion by shifting the mean of a biguassian distribution
def Nat_type3(R):
    X0 = []
    for j in range(len(R)):
        r1 = int(np.round(R[j] * 0.01 * NP_district))
        x = np.random.normal(mean, sd, NP_district)
        x = np.sort(x)
        d = max(x[1 : len(x)] - x[0 : len(x) - 1])
        while d > e_range[0]:
            x = np.random.normal(mean, sd, NP_district)
            x = np.sort(x)
            d = max(x[1 : len(x)] - x[0 : len(x) - 1])
        if r1 == 0:
            delta = -x[r1]
        elif r1 == NP_district:
            delta = -x[r1 - 1]
        else:
            delta = -(x[r1 - 1] + x[r1]) / 2
        x = x + delta
        X0.append(x)
    return X0


def natural_opinion(num_type):
    if num_type == 0:
        for nos in range(NOS):
            X0 = Nat_type1(House_of_rep)
            np.savez("DATA/NO_" + str(nos) + ".npz", X0=X0)
    elif num_type == 1:
        for nos in range(NOS):
            X0 = Nat_type2(House_of_rep)
            np.savez("DATA/NO_" + str(nos) + ".npz", X0=X0)
    elif num_type == 2:
        for nos in range(NOS):
            X0 = Nat_type3(House_of_rep)
            np.savez("DATA/NO_" + str(nos) + ".npz", X0=X0)


def winner(P, N):
    if P >= N:
        f = 1
    else:
        f = 0
    return f


# Function to influence the agents
def influence(x0, y, Mat_inv, num_agents_to_inf, Pn, Nn):
    flag = 0
    if Nn > Pn:
        flag = 1
        x0 = -x0
        Pn, Nn = Nn, Pn
        y = -y

    z = np.argsort(x0 + 1e2 * (x0 < 0) * (-x0))
    inf_array = np.zeros(len(x0))
    for i in range(num_agents_to_inf):
        inf_array[z[i]] = val_w0

    y_after_inf = y + np.matmul(Mat_inv, inf_array)

    p = np.sum(y_after_inf > 0)
    n = NP_district - p

    if flag == 1:
        p, n = n, p

    return p, n


# Function to calculate the outcome and to check if the applied budget changes the election outcome
def calc_HOR(h):
    row = M[h]
    num, eps = row
    num, eps = int(num), int(eps)
    data = np.load("DATA/NO_" + str(num) + ".npz")
    X = data["X0"]
    e = e_range[eps]
    success = np.zeros(
        (len(Influence_percentage_array), len(Districts_influenced))
    )  # Flag to determine whether the influence changed the outcome of election
    i = 0
    for g in Districts_influenced:
        x0 = np.array(X[g])
        A = (
            abs(np.tile(x0, (len(x0), 1)) - np.transpose(np.tile(x0, (len(x0), 1)))) < e
        ) * 1 - np.identity(len(x0))
        Mat_inv = np.linalg.inv(
            np.matmul(
                np.diag(np.array([1 / i if i != 0 else 0 for i in np.sum(A, axis=1)])),
                np.diag(np.sum(A, axis=1)) - A,
            )
            + np.identity(len(x0))
        )
        y = np.matmul(Mat_inv, x0)
        Pn = np.sum(y > 0)
        Nn = NP_district - Pn
        f1 = winner(Pn, Nn)
        for inf in range(len(Influence_percentage_array)):
            Inf_percentage = Influence_percentage_array[inf]
            num_agents_to_inf = int(np.round((Inf_percentage * 0.01) * NP_district))
            P, N = influence(x0, y, Mat_inv, num_agents_to_inf, Pn, Nn)
            f = winner(P, N)
            if f != f1:
                # print(g,i,'result changed')
                success[inf, i] = 1
            else:
                # print('No effect')
                success[inf, i] = 0
        i = i + 1
    # print('Success:',np.sum(success,axis=1))
    np.savez("DATA/USdatares_" + str(h) + ".npz", num=num, eps=eps, success=success)


# Function to parallelize the code and to collect the data to a single file
def HOR_ELECTION():
    pool = mp.Pool(mp.cpu_count())
    pool.map(calc_HOR, range(len(M)))
    pool.close()

    SUCCESS = np.zeros(
        (NOS, neps, len(Influence_percentage_array), len(Districts_influenced))
    )

    for i in range(NOS * neps):
        data = np.load("DATA/USdatares_" + str(i) + ".npz")
        eps = data["eps"]
        num = data["num"]
        SUCCESS[num, eps] = data["success"]

    np.savez(
        "US_CONST_EVEN/usrepub_" + str(num_type + 1) + "_" + str(year) + ".npz",
        e_range=e_range,
        SUCCESS=SUCCESS,
        Influence_percentage_array=Influence_percentage_array,
    )


if __name__ == "__main__":
    NOS = 100  # Number of simulations
    Num_states = 50  # Number of states
    Num_districts = 435  # Number of districts
    neps = 25
    e_range = np.linspace(0.05, 0.8, neps)
    e_num = np.arange(0, neps, 1)
    val_w0 = -0.1
    bi_mean = 0.25
    mean = 0
    sd = 0.2
    M = []
    NP_district = 501  # Number of people in each district
    Num_districts_perstate = np.zeros((Num_states))
    Num_type = 3
    M = [(j, k) for j, k in itertools.product(range(0, NOS, 1), range(0, neps, 1))]

    Influence_percentage_array = np.array(
        (
            0.25,
            0.5,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            12,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            75,
            100,
        )
    )

    # data=np.load('Districts_changed.npz')
    Districts_influenced = np.arange(0, Num_districts, 1)
    os.mkdir("US_CONST_EVEN")
    for year in range(2012, 2022, 2):
        df = pd.read_csv(
            "HOR_DATA/Election" + str(year) + ".csv",
            converters={"ID": str, "CD_NUM": str, "STATE_FP": str},
        )
        House_of_rep = np.array(df["Republican"])
        for num_type in range(Num_type):
            os.mkdir("DATA")
            natural_opinion(num_type)
            HOR_ELECTION()  # House of representative elections in districts
            shutil.rmtree("DATA")

stop = timeit.default_timer()
print("Time: ", (stop - start) / (60 * 60), "hours")
