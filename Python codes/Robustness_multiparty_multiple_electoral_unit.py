#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:07:37 2023

@author: glory
"""

# Robustness of electoral system of multi-partite system to external attack for 15 synthetic countries with [3,15] seats with an average of 9 seats per state.
# Four different types of electoral systems (Single representatives (SR), Winner-takes-all representatives(WTAR), Proportional representatives(PR), Proportional Ranked- Choice Voting System)
# The maximum difference in votes between two parties is 10%.

import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import numpy as np
import timeit
import multiprocessing as mp
import os
import shutil
import itertools
import random

# Nat_opn_generator.py : Generates the natural opinion in the simplex with equal density and volume for each party.
# Strategies_w0.py : Defines the influence vector.

import Nat_opn_generator as OPN_GEN  # Python file to generate natural opinion

import Electoral_system_function as ELEC_FUNC  # Python file with different type of electoral systems

start = timeit.default_timer()


# Function to generate natural opinion for a given state
def Nat_type(Num_party, num_state):
    Req_per = np.random.RandomState().uniform(
        low=100 / Num_party - 5, high=100 / Num_party + 5, size=(Num_party)
    )
    Req_ppl = (
        ((Req_per / np.sum(Req_per)) * Num_ppl_states[num_state]).round(decimals=0)
    ).astype("int")

    while Num_ppl_states[num_state] != np.sum(Req_ppl):
        if Num_ppl_states[num_state] - np.sum(Req_ppl) < 0:
            Req_ppl[random.sample(range(0, Num_party), 1)[0]] -= 1
        else:
            Req_ppl[random.sample(range(0, Num_party), 1)[0]] += 1

    x0 = np.zeros((Num_ppl_states[num_state], Num_party))
    Num_ppl_count = 0
    for num_party in range(Num_party):
        x0[
            Num_ppl_count : Num_ppl_count + Req_ppl[num_party]
        ] = OPN_GEN.Random_generation_pos(
            Req_ppl[num_party], Num_party, (num_party + 1)
        )
        Num_ppl_count += Req_ppl[num_party]  # Increase the number of count

    return x0


# Function to generate the natural opinion for a synthetic country
def NATURALOPINION(num):
    X0 = []
    for num_state in range(Num_states):
        x0 = Nat_type(Num_party, num_state)
        X0.append(x0)
    np.savez(
        "MULTIPARTY_NO/NO_" + str(Num_party) + "_" + str(num) + ".npz",
        x0=np.array(X0, dtype=object),
        Num_ppl_states=Num_ppl_states,
    )


# Function to determine the final outcome of the agents and source code for different types of electoral systems (PR,SR,WTAR,PRCV)
def outcome(h):
    num, eps = int(M[h][0]), int(M[h][1])
    data = np.load(
        "MULTIPARTY_NO/NO_" + str(Num_party) + "_" + str(num) + ".npz",
        allow_pickle=True,
    )
    X0 = data["x0"]
    e = e_range[eps]
    NUM_VOTES = np.zeros((Num_states, Num_party))
    NUM_VOTES_INITIAL = np.zeros((Num_states, Num_party))
    MAT_INV = []
    Result = []
    ADJ = []
    for num_state in range(Num_states):
        x0 = X0[num_state]
        NUM_VOTES_INITIAL[num_state] = np.sum(
            (np.argsort(x0).argsort() == Num_party - 1) * 1, axis=0
        )
        A = np.zeros((len(x0), len(x0)))
        for i in range(len(x0)):
            A[i, :] = (
                (np.linalg.norm(np.tile(x0[i], (len(x0), 1)) - x0, ord=1, axis=1)) < e
            ) * 1
        A = A - np.identity(len(x0))
        Mat_inv = np.linalg.inv(
            np.matmul(
                np.diag(np.array([1 / i if i != 0 else 0 for i in np.sum(A, axis=1)])),
                np.diag(np.sum(A, axis=1)) - A,
            )
            + np.identity(len(x0))
        )
        MAT_INV.append(Mat_inv)
        ADJ.append(A)
        Y = np.matmul(Mat_inv, x0)
        Result.append(Y)
        NUM_VOTES[num_state] = np.sum(
            (np.argsort(Y).argsort() == Num_party - 1) * 1, axis=0
        )

    np.savez(
        "MULTIPARTY_RES/Inital_res_" + str(h) + "_.npz",
        Num_party=Num_party,
        Num_states=Num_states,
        Num_seats_per_state=Num_seats_per_state,
        Num_districts=Num_districts,
        Num_ppl_states=Num_ppl_states,
        X0=np.array(X0, dtype=object),
        Result=np.array(Result, dtype=object),
        NUM_VOTES=np.array(NUM_VOTES, dtype=object),
    )

    ELEC_FUNC.Single_representative(h, MAT_INV)

    ELEC_FUNC.WTAR_representative(h, MAT_INV)

    ELEC_FUNC.Prop_representative(h, MAT_INV)

    ELEC_FUNC.RCV_system(h, MAT_INV)


# Function to sort and save the results in a compact form
def ELECTION_NEC():
    os.mkdir("MULTIPARTY_RES")
    os.mkdir("MULTIPARTY_SR")
    os.mkdir("MULTIPARTY_PR")
    os.mkdir("MULTIPARTY_WTAR")
    os.mkdir("MULTIPARTY_RCV")

    pool = mp.Pool(int(mp.cpu_count()))
    pool.map(outcome, range(len(M)))
    pool.close()

    EFF_SR = np.zeros((Num_simulation, neps))
    EFF_PR = np.zeros((Num_simulation, neps))
    EFF_WTAR = np.zeros((Num_simulation, neps))
    EFF_RCV = np.zeros((Num_simulation, neps))

    NUM_AGENTS_INFLUENCED_SR = np.zeros((Num_simulation, neps))
    NUM_AGENTS_INFLUENCED_PR = np.zeros((Num_simulation, neps))
    NUM_AGENTS_INFLUENCED_WTAR = np.zeros((Num_simulation, neps))
    NUM_AGENTS_INFLUENCED_RCV = np.zeros((Num_simulation, neps))

    for h in range(Num_simulation * neps):
        data = np.load(
            "MULTIPARTY_SR/Multipleparty_" + str(h) + ".npz", allow_pickle=True
        )
        num, eps = int(M[h][0]), int(M[h][1])
        EFF_SR[num, eps] = data["Effort"][0]
        NUM_AGENTS_INFLUENCED_SR[num, eps] = data["NUM_AGENTS_INFLUENCED"][0]

        data = np.load(
            "MULTIPARTY_PR/Multipleparty_" + str(h) + ".npz", allow_pickle=True
        )
        EFF_PR[num, eps] = data["Effort"][0]
        NUM_AGENTS_INFLUENCED_PR[num, eps] = data["NUM_AGENTS_INFLUENCED"][0]

        data = np.load(
            "MULTIPARTY_WTAR/Multipleparty_" + str(h) + ".npz", allow_pickle=True
        )
        EFF_WTAR[num, eps] = data["Effort"][0]
        NUM_AGENTS_INFLUENCED_WTAR[num, eps] = data["NUM_AGENTS_INFLUENCED"][0]

        data = np.load(
            "MULTIPARTY_RCV/Multipleparty_" + str(h) + ".npz", allow_pickle=True
        )
        EFF_RCV[num, eps] = data["Effort"][0]
        NUM_AGENTS_INFLUENCED_RCV[num, eps] = data["NUM_AGENTS_INFLUENCED"][0]

    np.savez(
        "Multi_States/Pol_Multiparty_diff_elec_sys_"
        + str(Num_party)
        + "_"
        + str(N_S)
        + ".npz",
        Num_simulation=Num_simulation,
        e_range=e_range,
        EFF_SR=EFF_SR,
        EFF_PR=EFF_PR,
        EFF_WTAR=EFF_WTAR,
        EFF_RCV=EFF_RCV,
        NUM_AGENTS_INFLUENCED_SR=NUM_AGENTS_INFLUENCED_SR,
        NUM_AGENTS_INFLUENCED_PR=NUM_AGENTS_INFLUENCED_PR,
        NUM_AGENTS_INFLUENCED_WTAR=NUM_AGENTS_INFLUENCED_WTAR,
        NUM_AGENTS_INFLUENCED_RCV=NUM_AGENTS_INFLUENCED_RCV,
    )

    shutil.rmtree("MULTIPARTY_RES")
    shutil.rmtree("MULTIPARTY_SR")
    shutil.rmtree("MULTIPARTY_PR")
    shutil.rmtree("MULTIPARTY_WTAR")
    shutil.rmtree("MULTIPARTY_RCV")


if __name__ == "__main__":
    Num_simulation = 100  # Number of simulations
    Min_agents = 101  # Number agents corresponding to one seat
    neps = 51
    e_range = np.linspace(0, 2, neps).round(
        decimals=3
    )  # Confidence bound parameter range
    M = [
        (j, k)
        for j, k in itertools.product(range(0, Num_simulation, 1), range(0, neps, 1))
    ]

    # Number of parties range
    Num_party_start = 3
    Num_party_end = 8

    data = np.load("Multi_States/Initial_considerations.npz", allow_pickle=True)
    NUM_STATES = data["NUM_STATES"]
    NUM_SEATS_PER_STATE = data["NUM_SEATS_PER_STATE"]

    np.savez(
        "Multi_States/Initial_considerations.npz",
        NUM_STATES=NUM_STATES,
        NUM_SEATS_PER_STATE=NUM_SEATS_PER_STATE,
        Num_simulation=Num_simulation,
        neps=neps,
        e_range=e_range,
        Num_party_start=Num_party_start,
        Num_party_end=Num_party_end,
    )

    for N_S in range(len(NUM_STATES)):
        Num_states = NUM_STATES[N_S]  # Number of states in the synthetic countries
        Num_seats_per_state = NUM_SEATS_PER_STATE[
            N_S
        ]  # Array of number of seats per state

        Num_districts = int(
            np.sum(NUM_SEATS_PER_STATE[N_S])
        )  # Total number of seats which is assumed to be equal to the number of districts
        Num_ppl_states = (
            Min_agents * Num_seats_per_state
        )  # Number of people in each state
        Num_ppl_states[
            Num_ppl_states % 2 == 0
        ] += 1  # Increasing the number of people if it's even

        # Generating the natural opinion
        os.mkdir("MULTIPARTY_NO")

        for Num_party in range(Num_party_start, Num_party_end):
            # Paralleling the generation of natural opinion
            pool = mp.Pool(int(mp.cpu_count()))
            pool.map(NATURALOPINION, range(Num_simulation))
            pool.close()

        for Num_party in range(Num_party_start, Num_party_end):
            # Evaluating the outcome of the election and to compute the effort needed to change the election's outcome
            ELECTION_NEC()

        shutil.rmtree("MULTIPARTY_NO")

stop = timeit.default_timer()
print("Time:", (stop - start) / (60 * 60))
