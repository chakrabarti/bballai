import numpy as np
import pandas as pd
from game_calc import makeMatrixWithAggression, nash_lp


team_1 = ['Stephen Curry', 'Klay Thompson', 'Draymond Green', 'Kevin Durant', 'Zaza Pachulia']
team_2 = ['James Harden', 'Chris Paul', 'Trevor Ariza', 'Clint Capela', 'Nene']

m = makeMatrixWithAggression(1,team_1,team_2,team_1[0])
m2 = makeMatrixWithAggression(2, team_2, team_1,team_2[0])

mdf = pd.DataFrame(m)
m2df = pd.DataFrame(m2)

print(mdf)
print(m2df)

def add_noise_to_matrix(A, variance = 0.05):
    gamevals = A.values
    noise = np.random.normal(0, variance, gamevals.shape)
    return gamevals + noise   

def get_noisy_nash(mdf):
    gamevals = mdf.values

    strats = dict()

    last_off, k_off = nash_lp(-1*gamevals.T)
    last_def, k_def = nash_lp(gamevals)
    numiters = 100
    for x in range(1, numiters):
        print("iteration " + str(x))
        gamevals = add_noise_to_matrix(mdf)
        # gamevals = mdf.values
        # noise = np.random.normal(0, 1/20, gamevals.shape)
        # gamevals += noise
        b_off, k_off = nash_lp(-1*gamevals.T) 
        b_def, k_def = nash_lp(gamevals)
        last_off += b_off
        last_def += b_def

    last_off = last_off/numiters
    last_def = last_def/numiters
    thing_off = {}
    thing_def = {}

    summ_off = 0
    summ_def = 0
    colnames = list(mdf.columns.values)

    good_off = np.where(last_off >= 0.01)
    bad_off = np.where(last_off < 0.01)

    last_off[bad_off] = 0
    last_off[good_off] = last_off[good_off]/np.sum(last_off[good_off])

    good_def = np.where(last_def >= 0.01)
    bad_def = np.where(last_def < 0.01)

    last_def[bad_def] = 0
    last_def[good_def] = last_def[good_def]/np.sum(last_def[good_def])

    # for i,p in enumerate(last_off):
    #     if(p > 0.05):
    #         summ_off += p
    #         thing_off[mdf.index[i]] = p
    # for i,p in enumerate(last_def):
    #     if(p > 0.05):
    #         summ_def += p
    #         thing_def[colnames[i]] = p
    # return thing_def,thing_off

    # for i,p in enumerate(last_off):
    #     thing_off[mdf.index[i]] = p
    # for i,p in enumerate(last_def):
    #     thing_def[colnames[i]] = p

    # # return thing_def,thing_off

    return last_def, last_off

#print(get_noisy_nash(mdf))

deff,offf = get_noisy_nash(mdf)
# deff2,offf2 = get_noisy_nash(m2df)

def sample_strategy(mixed):
    opts = np.arange(len(mixed))
    return np.random.choice(opts, p = mixed)



p = sample_strategy(deff)
p2 = sample_strategy(offf)
print(mdf.columns.values[p])
print(mdf.index[p2])

old_strat_def = def_strat = deff
old_strat_off = off_strat = offf

# def_strat = None
# off_strat = None

w = 0.2
v = 0.7
print(deff)
print(offf)
num_poss = 10
for poss in range(num_poss):
    current_def = sample_strategy(def_strat)
    current_off = sample_strategy(off_strat)

    print("Play during possession {}".format(poss))
    print("\t" + str(mdf.columns.values[current_def]))
    print("\t" + str(mdf.index[current_off]))

    add_def = np.zeros(len(old_strat_def))
    add_def[current_def] = 1
    old_strat_def = w*old_strat_def + (1-w)*add_def

    add_off = np.zeros(len(old_strat_off))
    add_off[current_off] = 1
    old_strat_off = w*old_strat_off + (1-w)*add_off

    gamevals = add_noise_to_matrix(mdf)
    best_response_off = np.zeros(len(old_strat_off))
    best_response_off[np.argmax(gamevals.dot(old_strat_def))] = 1

    best_response_def = np.zeros(len(old_strat_def))
    best_response_def[np.argmin((old_strat_off.T).dot(gamevals))] = 1

    def_strat = v*def_strat + (1-v)*best_response_def
    off_strat = v*off_strat + (1-v)*best_response_off
    