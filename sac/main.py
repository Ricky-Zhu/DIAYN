from sac import SACAgent
from discriminator import SkillDiscriminator
from buffer import ReplayBuffer
import numpy as np
import torch
import wandb


def get_one_hot_encode_skill(skill_nums):
    skill = np.random.randint(skill_nums)
    skill_one_hot = np.zeros(skill_nums)
    skill_one_hot[skill] = 1
    return skill_one_hot


def train_loop(env, args):
    wandb.login()
    wandb.init(
        project="DIAYN_{}".format(args.env),
        config=vars(args)
    )

    agent = SACAgent(env, args)
    buffer = ReplayBuffer(env.observation_space.shape[0],
                          env.action_space.shape[0],
                          args.buffer_size,
                          args.skill_nums,
                          args.device)
    discriminator = SkillDiscriminator(obs_dim=env.observation_space.shape[0],
                                       skill_nums=args.skill_nums,
                                       hidden_size=args.hidden_size,
                                       lr=args.d_lr,
                                       device=args.device).to(args.device)

    total_interaction_steps = 0

    for epoch in range(args.total_epochs):
        for ep in range(args.episodes_per_epoch):
            o = env.reset()

            # sample the skill, as the skill dist is uniform to ensure max entropy
            # make the skill one-hot
            z = get_one_hot_encode_skill(args.skill_nums)

            for i in range(args.max_episode_length):
                a = agent.get_action(o, z)
                o2, r, d, _ = env.step(a)  # the reward here is not going to be used
                total_interaction_steps += 1

                buffer.store(o, a, r, o2, d, z)
                o = o2
                if d:
                    break

            for _ in range(args.update_cycles):
                data = buffer.sample_batch(batch_size=args.batch_size)

                # get the reward given in DIAYN
                d_loss, rewards = discriminator.update(data)

                # replace the rewards part in data
                data['rew'] = rewards
                loss_pi, loss_q1, loss_q2 = agent.update(data)

                wandb.log({'discriminator loss': d_loss,
                           'actor loss': loss_pi,
                           'q1 loss': loss_q1,
                           'q2 loss': loss_q2}, step=total_interaction_steps)

        agent.save_model()
        discriminator.save_model()


if __name__ == "__main__":
    import argparse
    import gym
    from goal_env.mujoco import *
    from wrapper import WrapperDictEnv

    parser = argparse.ArgumentParser()

    # env parameters
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--env', type=str, default='AntEmpty-v0')

    # agent parameters
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--polyak', type=float, default=0.005)
    parser.add_argument('--q-lr', type=float, default=3e-4)
    parser.add_argument('--p-lr', type=float, default=3e-4)
    parser.add_argument('--reward-scale', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')

    # discriminator params
    parser.add_argument('--skill-nums', type=int, default=20)
    parser.add_argument('--d-lr', type=float, default=3e-4)

    # buffer parameters
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-size', type=int, default=int(1e6))

    # training parameters
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--seed', '-s', type=int, default=44)
    parser.add_argument('--total-epochs', type=int, default=300)
    parser.add_argument('--episodes-per-epoch', type=int, default=1)
    parser.add_argument('--initialize-buffer-steps', type=int, default=10000)
    parser.add_argument('--max-episode-length', type=int, default=1000)
    parser.add_argument('--update-cycles', type=int, default=1000)
    parser.add_argument('--evaluate-interval-epochs', type=int, default=5)
    parser.add_argument('--evaluation-nums', type=int, default=5)
    args = parser.parse_args()

    ##########################################################################
    env = gym.make(args.env)
    env = WrapperDictEnv(env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)  # to ensure during the early random exploration the data the same

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loop(env, args)
