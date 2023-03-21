import argparse
import gym
from util import record, prune_and_cap_trajs, compute_hist
from ma_gym.wrappers import Monitor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Agent for ma-gym')
    parser.add_argument('--env', default='PredatorPrey5x5-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='episodes (default: %(default)s)')
    args = parser.parse_args()

    print('Enter the actions space together and press enter ( Eg: \'11<enter>\' which meanes take 1'
          ' for agent 1 and 1 for agent 2)')

    trajs = []
    env = gym.make('ma_gym:' + args.env)
    env = Monitor(env, directory=args.env + '/monitor', force=True)
    print([i for i in range(env.env.action_space[1].n)])
    print(env.get_action_meanings(1))

    for ep_i in range(args.episodes):
        traj = []
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0
        env.seed(ep_i)
        obs_n = env.reset()
        env.render()

        while not all(done_n):
            action_n = [int(_) for _ in input('Action:')]
            traj += [{"obs_n": [tuple(obs_n[i][0:2]) for i in range(env.n_agents)], "action_n": action_n}]
            obs_n, reward_n, done_n, info = env.step(action_n)
            ep_reward += sum(reward_n)
            env.render()

        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
        traj = {k: v for k, v in enumerate(traj)}
        trajs += [traj]

    env.close()

    trajs = {k: v for k, v in enumerate(trajs)}
    trajs_capped, max_horizon = prune_and_cap_trajs(trajs)
    hist_all, hist_zero, hist_trans = compute_hist(trajs_capped, env.n_agents, max_horizon)

    # record all data
    record([env, trajs, trajs_capped, max_horizon, hist_all, hist_zero, hist_trans], directory=args.env, label='interactive_data_'+str(args.episodes))
    print("\nSaved python data to `data/"+args.env+'interactive_data_'+str(args.episodes)+".pickle`")