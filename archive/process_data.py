import pprint
from util import load, record

# argparse
import argparse
parser = argparse.ArgumentParser(description='Random Agent for ma-gym')
parser.add_argument('--env', default='PredatorPrey5x5-v0',
                    help='Name of the environment (default: %(default)s)')
parser.add_argument('--episodes', type=int, default=1,
                    help='episodes (default: %(default)s)')
args = parser.parse_args()

def compute_hist(trajs_capped, n_agents, max_horizon):
    print("Computed histograms of joint probs, init state dist, transition.")

    # compute hist for reference joint probability dist of state-action pairs
    hist_all = {}
    for i in range(n_agents):
        hist_t = {}
        for t in range(max_horizon):
            list = [(trajs_capped[ep][t]['action_n'][i], trajs_capped[ep][t]['obs_n'][i]) for ep in range(len(trajs_capped))]
            freq = {}
            for l in list:
                if l not in freq.keys():
                    freq[l] = 1
                else:
                    freq[l] += 1
            hist = {k: v/len(trajs_capped) for k, v in freq.items()}
            hist_t[t] = hist
        hist_all[i] = hist_t
    
    # compute initial state distribution
    hist_zero = {}
    for i in range(n_agents):
        total_i = 0
        count_zero_i = {}
        for ep in range(len(trajs_capped)):
            s = trajs_capped[ep][0]['obs_n'][i]
            if s not in count_zero_i.keys():
                count_zero_i[s] = 1
            else:
                count_zero_i[s] += 1
            total_i += 1
        hist_zero[i] = {k: v/total_i for k,v in count_zero_i.items()}

    # compute transition probabilities
    hist_trans={}
    for i in range(n_agents):
        count_sas = {}
        count_sa = {}
        for ep in range(len(trajs_capped)):
            for t in range(max_horizon-1):
                s = trajs_capped[ep][t]['obs_n'][i]
                a = trajs_capped[ep][t]['action_n'][i]
                s_next = trajs_capped[ep][t+1]['obs_n'][i]

                if (s,a) not in count_sa.keys():
                    count_sa[(s,a)] = 1
                else:
                    count_sa[(s,a)] += 1
                
                if (s,a,s_next) not in count_sas.keys():
                    count_sas[(s,a,s_next)] = 1
                else:
                    count_sas[(s,a,s_next)] += 1
        hist_trans[i] = {(s,a,s_next): v / count_sa[(s,a)] for (s,a,s_next),v in count_sas.items()}

    return hist_all, hist_zero, hist_trans


# import trajs_capped with k trajectories, each with a horizon of max_horizon
env, trajs, trajs_capped, max_horizon = load(args.env+'/data_'+str(args.episodes)+'.pickle')
print('length of trajs', max_horizon)
print('number of trajs', len(trajs_capped))

# compute histogram of state-action pairs for each player i at each timestep t
hist_all, hist_zero, hist_trans = compute_hist(trajs_capped, env.n_agents, max_horizon)
# pprint.pprint(len(hist_all))
# pprint.pprint(hist_zero)

# record processed data
record([hist_all, hist_zero, hist_trans], directory=args.env, label='processed_data_'+str(args.episodes))