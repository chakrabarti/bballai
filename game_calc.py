from nba_api.stats.static import teams
from nba_api.stats.endpoints import playervsplayer
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo,shotchartdetail, leaguedashplayerstats, playerdashptshotdefend, commonplayerinfo,playerdashptpass

import pandas as pd
from itertools import permutations 
from joblib import Memory
import time
from multiprocessing import Process 
import nashpy as nash
import cvxpy as cp
import numpy as np

custom_headers = {
    'Host': 'stats.nba.com',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}
cachedir = '/users/rishav/documents/15-780/project'
memory = Memory(cachedir, verbose=0)

@memory.cache
def getPlayerId(name):
    player = players.find_players_by_full_name(name)[0]
    pid = player['id']
    return pid

@memory.cache
def playerOnPlayer(offname,defname,season='2017-18'):
    offplayerID = getPlayerId(offname)
    defplayerID = getPlayerId(defname)
    playvsplay = playervsplayer.PlayerVsPlayer(player_id = offplayerID,vs_player_id = defplayerID,season=season,headers=custom_headers)
    return playvsplay.data_sets[6].get_data_frame()

@memory.cache
def getPlayerShotChart(playername,season='2017-18'):
    offplayerID = getPlayerId(playername)
    playvsplay = playervsplayer.PlayerVsPlayer(player_id = offplayerID,vs_player_id = offplayerID,season=season,headers=custom_headers)
    return playvsplay.data_sets[5].get_data_frame()

@memory.cache
def getPlayerTeamID(name):
    playid = getPlayerId(name)
    obj = commonplayerinfo.CommonPlayerInfo(player_id=playid)
    print(obj.data_sets[0].get_data_frame()["TEAM_ABBREVIATION"].iloc[0])
    tdict = teams.find_team_by_abbreviation(obj.data_sets[0].get_data_frame()["TEAM_ABBREVIATION"].iloc[0])
    print(tdict["id"])
    return tdict["id"]

#print(getPlayerShotChart('klay thompson'))
#print(playerOnPlayer('Zaza Pachulia', 'paul george'))

OFFENSE = ['Stephen Curry', 'Klay Thompson', 'Kevin Durant', 'Zaza Pachulia', 'Draymond Green']
DEFENSE = ['James Harden', 'Chris Paul', 'Trevor Ariza', 'Clint Capela', 'Nene']
Ball_handler = OFFENSE[2]
SCORING = { 'Restricted Area' : 2, 'In The Paint (Non-RA)' : 2, 'Mid-Range': 2, 'Left Corner 3': 3, 'Right Corner 3' : 3, 'Above the Break 3' : 3, 'Backcourt' : 3}
AREATODEFENSE = {'Restricted Area' : "Less Than 6 Ft", 'In The Paint (Non-RA)': "Less Than 10 Ft", "Mid-Range":"Greater Than 15 Ft", 'Left Corner 3' : "3 Pointers",'Right Corner 3': "3 Pointers", 'Above the Break 3':"3 Pointers", 'Backcourt' : "3 Pointers"  }

@memory.cache
def playerDefense(name):
    teamid = getPlayerTeamID(name)
    #print(teamid)
    pid = getPlayerId(name)
    p = playerdashptshotdefend.PlayerDashPtShotDefend(player_id=pid,team_id=0,season="2017-18",per_mode_simple="Totals")
    print(p.data_sets[0].get_data_frame())
    return p.data_sets[0].get_data_frame()

@memory.cache
def playerPassing(name):
    #print(teamid)
    pid = getPlayerId(name)
    p = playerdashptpass.PlayerDashPtPass(player_id=pid,team_id=0,season="2017-18",per_mode_simple="PerGame")
    print(p.data_sets[0].get_data_frame()['PASS'])
    return p.data_sets[0].get_data_frame()

@memory.cache
def playerPassToOther(name,pass_target):
    df = playerPassing(name)
    targetId = getPlayerId(pass_target)
    rowTOwork = df[df["PASS_TEAMMATE_PLAYER_ID"] == targetId].iloc[0]
    asts = rowTOwork["AST"]
    passes = rowTOwork["PASS"]
    return(asts/passes)
    #print(asts,passes)

@memory.cache
def calculateMatrixEntry(offp,defp,matchup,player,zone):
    
    SCORING = { 'Restricted Area' : 2, 'In The Paint (Non-RA)' : 2, 'Mid-Range': 2, 'Left Corner 3': 3, 'Right Corner 3' : 3, 'Above the Break 3' : 3, 'Backcourt' : 3}

    playind = offp.index(player)
    opp = defp[matchup[playind]]
    player_shot = getPlayerShotChart(player)
    play_on_player = playerOnPlayer(player, opp)
    defensePlayer = playerDefense(opp)
    areaForDef = AREATODEFENSE[zone]

    areaForDefProd = defensePlayer[defensePlayer['DEFENSE_CATEGORY'] == areaForDef].iloc[0]['D_FG_PCT']
    print(areaForDefProd)

    player_shot_row_overall = player_shot[player_shot['GROUP_VALUE'] == zone].iloc[0]
    try:
        vs_player_row = play_on_player[play_on_player['GROUP_VALUE'] == zone].iloc[0]
    except:
        vs_player_row = play_on_player[play_on_player['GROUP_VALUE'] == zone]


    overall_FGA = float(player_shot_row_overall['FGA'])
    overall_FGM = float(player_shot_row_overall['FGM'])
    if(not vs_player_row.empty):
        def_FGA = float(vs_player_row['FGA'])
        def_FGM = float(vs_player_row['FGM'])
    else:
        def_FGA = 0.0
        def_FGM = 0.0
    
    
    try: our_prob = (1 + (overall_FGM/overall_FGA))*(1+(def_FGM/def_FGA))*(1+(areaForDefProd))
    except: our_prob = 0
        
    expectedoutput = our_prob*SCORING[zone]
    #print("not cached")
    return expectedoutput

def pairWiseShotChartCache(teama, teamb):
    for player1 in teama:
        for player2 in teamb:
            a = time.time()
            if(player1 == player2): continue
            print(player1 + " VS " + player2)
            try: playerOnPlayer(player1,player2)
            except:
                print("this didnt work") 
                continue
            try: playerOnPlayer(player2,player1)
            except:
                print("this didnt work")
                continue
            b = time.time()
            print(b-a)



#calculateMatrixEntry(OFFENSE,DEFENSE,[0,1,2,3,4], 'Zaza Pachulia', 'Restricted Area')




def setPlayerShotChartCache():
    playerlist = leaguedashplayerstats.LeagueDashPlayerStats(season='2017-18')
    #print(playerlist.data_sets[0].get_dict()['headers'])
    #print(playerlist.data_sets[0].get_dict()['data'])#["CommonAllPlayers"])
    for player in playerlist.data_sets[0].get_dict()['data']:
        print(player[1])
        try: getPlayerShotChart(player[1])
        except:
            print("name doesnt exist") 
            continue

@memory.cache
def nash_lp(A):
    q = cp.Variable(A.shape[1])
    v = cp.Variable(1)
    constraints = []
    constraints += [cp.sum(q) == 1]
    constraints += [q >= 0]
    constraints += [A*q <= v]
    obj = cp.Minimize(v)
    prob = cp.Problem(obj, constraints)
    result = prob.solve(solver='ECOS')
    print(np.around(q.value,decimals=3), result)
    return np.around(q.value,decimals=3), result

@memory.cache
def stackelberg(u1, u2):
    '''
    Compute the optimal Stackelberg strategy for the given 2-player normal form game.

    Arguments:
        u1: a 2D numpy array representing the utility function for player 1
        u2: a 2D numpy array representing the utility function for player 2

    Return:
        (u,x1,s2) where
            u is the expected utility for player 1 of mixed strategy x1 and s2
            x1 is a 1D numpy array of probabilities representing the mixed strategy of player 1
            s2 is the index (0-based) of the pure strategy for player 2
    '''

    n1, n2 = u1.shape

    results = np.zeros(n2)
    xvals = np.zeros((n2, n1))

    bestval = np.zeros(n1)
    bestresult = None
    bests2 = None

    for s2 in range(n2):
        x= cp.Variable(n1)

        constraints = [x <= 1, x >= 0]
        constraints += [cp.sum(x) == 1]

        for s2prime in range(n2):
            constraints += [cp.sum(cp.multiply(x, u2[:, s2])) >= cp.sum(cp.multiply(x, u2[:, s2prime]))]

        objective = cp.Maximize(cp.sum(cp.multiply(x, u1[:, s2])))
    
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver='ECOS') 
        results[s2] = result
        xvals[s2] = x.value

        if not bestresult or result > (bestresult + 1e-5):
            bestval = np.array(x.value)
            bestresult = result
            bests2 = s2


    # TODO: These are placeholder values that have the correct type and shape.
    #       Replace these with your implementation.
    # best_util = -1
    # best_x1 = np.zeros(n1)
    # best_s2 = -1

    best_util = bestresult
    best_x1 = bestval
    best_s2 = bests2

    return (best_util, np.around(np.array(best_x1),decimals=6), best_s2)

l = list(permutations(range(0,5)))
matrix = {}
pairWiseShotChartCache(OFFENSE,DEFENSE)
PASS_WEIGHT = { 'Restricted Area' : .45, 'In The Paint (Non-RA)' : 0.55, 'Mid-Range': 0.75, 'Left Corner 3': 0.85, 'Right Corner 3' : 0.85, 'Above the Break 3' : 1, 'Backcourt' : -1}
for matchup in l:
    if(not ((matchup[0] == 1 and matchup[1] == 0) or (matchup[0] == 0 and matchup[1] == 1))): continue
    matrix[str(matchup)] = {}
    for player in OFFENSE:
        for zone in SCORING:
            
            print(str(matchup) + " " + player + " " + zone)
            output = calculateMatrixEntry(OFFENSE,DEFENSE,matchup, player, zone)
            if(player == Ball_handler):
                matrix[str(matchup)][(str(player + " "+ zone))] = output*min(PASS_WEIGHT[zone]+0.2,1)
            else:
                pass_cost = playerPassToOther(Ball_handler,player)
                matrix[str(matchup)][("Pass to " + str(player + " "+ zone))] = output*0.9*PASS_WEIGHT[zone]*pass_cost
            print(output)

#p = playerdashptshotdefend.PlayerDashPtShotDefend(player_id=0,team_id=0,season="2017-18",per_mode_simple="Totals")
#print(p.data_sets[0].get_data_frame().groupby("DEFENSE_CATEGORY"))

#gb = p.data_sets[0].get_data_frame().groupby("DEFENSE_CATEGORY")['D_FG_PCT'].mean()
#print(gb)
#playerDefense("Lebron James")
#playerDefense("Klay Thompson")
#playerDefense("James Harden")
print(pd.DataFrame(matrix))
matDf = pd.DataFrame(matrix) 
gamevals = matDf.values
#print(matDf)

#p = nash_lp(gamevals)
#print(p)

#p = nash_lp(gamevals.T)
#print(p)

a,b,c = stackelberg(gamevals,(gamevals*-1))
print(a,b,c)

for i,p in enumerate(b):
    if(p > 0):
        print(matDf.index[i] + " with probability " + str(p))

#print(playerPassToOther('Stephen Curry',"Kevin Durant"))
#setPlayerShotChartCache()
#getPlayerShotChart("Klay Thompson")