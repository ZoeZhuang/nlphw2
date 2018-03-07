import numpy as np



#   hidden states
hidden_state = ['start','verb', 'noun','adv','end']

#   obsevition seqs
obsevition = ['beginning','learning', 'changes', 'throughly','stop']


#   print the final max probability matrix
#   return the best path according to the @obs, @states, @start_p, @trans_p, @emit_p
def compute(obs, states, start_p, trans_p, emit_p):
    #   the final max probability matrix
    max_p = np.zeros((len(obs), len(states)))

    #   the path matrix
    path = np.zeros((len(states), len(obs)))

    #   initialization
    for i in range(len(states)):
        max_p[0][i] = start_p[i] * emit_p[i][obs[0]]
        path[i][0] = i

    for t in range(1, len(obs)):
        newpath = np.zeros((len(states), len(obs)))
        for y in range(len(states)):
            prob = -1
            for y0 in range(len(states)):
                #fomula
                nprob = max_p[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]]

                if nprob > prob:
                    prob = nprob
                    state = y0
                    #   record the path
                    max_p[t][y] = prob
                    for m in range(t):
                        newpath[y][m] = path[state][m]
                    newpath[y][t] = y

        path = newpath
    print("**print the final max probability matrix:**")
    for kk in range(len(obs)):
        for tt in range(len(states)):
            print('%2E' % max_p[kk][tt],end='    ')
        print('\n')
    max_prob = -1
    path_state = 0
    #   return the best path
    for y in range(len(states)):
        if max_p[len(obs)-1][y] > max_prob:
            max_prob = max_p[len(obs)-1][y]
            path_state = y

    return path[path_state]

state_s = [0, 1, 2, 3, 4]
obser = [0, 1, 2, 3,4]

#   probability of initial state
start_probability = [1, 0 ,0 ,0 ,0]

#   transititon_probability
transititon_probability = np.array([[0,0.3,0.2,0,0],
                                    [0,0.1,0.4,0.4,0],
                                    [0,0.3,0.1,0.1,0],
                                    [0,0,0,0,0.1],
                                    [0,0,0,0,0]])

#   emission_probability
emission_probability = np.array([[1,0,0,0,0],
                                 [0,0.003,0.004,0,0],
                                 [0,0.001,0.003,0,0],
                                 [0,0,0,0.002,0],
                                 [1,1,1,1,1]])

result = compute(obser, state_s, start_probability, transititon_probability, emission_probability)


#print out the the tags for 'beginning','learning', 'changes', 'throughly','stop', P.S. the 'beginning' and 'stop' are in
# the sentence.
print("**print out the tags for the sentence including the 'beginning' and 'stop':**")
for k in range(len(result)):
    print(hidden_state[int(result[k])],end='    ')