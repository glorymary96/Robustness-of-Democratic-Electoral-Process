#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:08:47 2023

@author: glory
"""
import numpy as np


def Strategies(Num_party, winner_to_be):
    w0 = np.zeros((Num_party))
    if Num_party == 2:
        w0 = np.zeros((Num_party))
        if winner_to_be == 1:
            w0[winner_to_be], w0[0] = 1, -1
        else:
            w0[winner_to_be], w0[1] = 1, -1

    elif Num_party == 3:
        if winner_to_be == 1:
            w0[winner_to_be], w0[0], w0[2] = 1, -0.5, -0.5
        elif winner_to_be == 0:
            w0[winner_to_be], w0[1], w0[2] = 1, -0.6, -0.4
        else:
            w0[winner_to_be], w0[1], w0[0] = 1, -0.6, -0.4

    elif Num_party == 4:
        if winner_to_be == 0:
            w0[winner_to_be], w0[1], w0[2], w0[3] = 1, -0.5, -0.3, -0.2
        elif winner_to_be == 1:
            w0[winner_to_be], w0[0], w0[2], w0[3] = 1, -0.4, -0.4, -0.2
        elif winner_to_be == 2:
            w0[winner_to_be], w0[0], w0[1], w0[3] = 1, -0.2, -0.4, -0.4
        else:
            w0[winner_to_be], w0[0], w0[1], w0[2] = 1, -0.2, -0.3, -0.5

    elif Num_party == 5:
        if winner_to_be == 0:
            w0[winner_to_be], w0[1], w0[2], w0[3], w0[4] = 1, -0.45, -0.3, -0.2, -0.05
        elif winner_to_be == 1:
            w0[winner_to_be], w0[0], w0[2], w0[3], w0[4] = 1, -0.35, -0.35, -0.2, -0.1
        elif winner_to_be == 2:
            w0[winner_to_be], w0[0], w0[1], w0[3], w0[4] = 1, -0.2, -0.3, -0.3, -0.2
        elif winner_to_be == 3:
            w0[winner_to_be], w0[0], w0[1], w0[2], w0[4] = 1, -0.1, -0.2, -0.35, -0.35
        else:
            w0[winner_to_be], w0[0], w0[1], w0[2], w0[3] = 1, -0.05, -0.2, -0.3, -0.45

    elif Num_party == 6:
        if winner_to_be == 0:
            w0[winner_to_be], w0[1], w0[2], w0[3], w0[4], w0[5] = (
                1,
                -0.4,
                -0.25,
                -0.2,
                -0.1,
                -0.05,
            )
        elif winner_to_be == 1:
            w0[winner_to_be], w0[0], w0[2], w0[3], w0[4], w0[5] = (
                1,
                -0.3,
                -0.3,
                -0.2,
                -0.15,
                -0.05,
            )
        elif winner_to_be == 2:
            w0[winner_to_be], w0[0], w0[1], w0[3], w0[4], w0[5] = (
                1,
                -0.2,
                -0.25,
                -0.25,
                -0.2,
                -0.1,
            )
        elif winner_to_be == 3:
            w0[winner_to_be], w0[0], w0[1], w0[2], w0[4], w0[5] = (
                1,
                -0.1,
                -0.2,
                -0.25,
                -0.25,
                -0.2,
            )
        elif winner_to_be == 4:
            w0[winner_to_be], w0[0], w0[1], w0[2], w0[3], w0[5] = (
                1,
                -0.05,
                -0.15,
                -0.2,
                -0.3,
                -0.3,
            )
        else:
            w0[winner_to_be], w0[0], w0[1], w0[2], w0[3], w0[4] = (
                1,
                -0.05,
                -0.1,
                -0.2,
                -0.25,
                -0.4,
            )

    elif Num_party == 7:
        if winner_to_be == 0:
            w0[winner_to_be], w0[1], w0[2], w0[3], w0[4], w0[5], w0[6] = (
                1,
                -0.4,
                -0.2,
                -0.15,
                -0.12,
                -0.08,
                -0.05,
            )
        elif winner_to_be == 1:
            w0[winner_to_be], w0[0], w0[2], w0[3], w0[4], w0[5], w0[6] = (
                1,
                -0.25,
                -0.25,
                -0.2,
                -0.15,
                -0.1,
                -0.05,
            )
        elif winner_to_be == 2:
            w0[winner_to_be], w0[0], w0[1], w0[3], w0[4], w0[5], w0[6] = (
                1,
                -0.2,
                -0.25,
                -0.25,
                -0.2,
                -0.07,
                -0.03,
            )
        elif winner_to_be == 3:
            w0[winner_to_be], w0[0], w0[1], w0[2], w0[4], w0[5], w0[6] = (
                1,
                -0.05,
                -0.15,
                -0.3,
                -0.3,
                -0.15,
                -0.05,
            )
        elif winner_to_be == 4:
            w0[winner_to_be], w0[0], w0[1], w0[2], w0[3], w0[5], w0[6] = (
                1,
                -0.03,
                -0.07,
                -0.2,
                -0.25,
                -0.25,
                -0.2,
            )
        elif winner_to_be == 5:
            w0[winner_to_be], w0[0], w0[1], w0[2], w0[3], w0[4], w0[6] = (
                1,
                -0.05,
                -0.1,
                -0.15,
                -0.2,
                -0.25,
                -0.25,
            )
        else:
            w0[winner_to_be], w0[0], w0[1], w0[2], w0[3], w0[4], w0[5] = (
                1,
                -0.05,
                -0.08,
                -0.12,
                -0.15,
                -0.2,
                -0.4,
            )

    return w0
