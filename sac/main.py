from sac import SACAgent
from discriminator import SkillDiscriminator
from buffer import ReplayBuffer
import numpy as np
import torch
import datetime
import cv2
import os
from tqdm import tqdm


def get_one_hot_encode_skill(skill_nums):
    skill = np.random.randint(skill_nums)
    skill_one_hot = np.zeros(skill_nums)
    skill_one_hot[skill] = 1
    return skill_one_hot, skill


def play(env, args):
    from mujoco_py import GlfwContext
    GlfwContext(offscreen=True)
    agent = SACAgent(env, args)
    agent.load_model()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # make the dir to store the videos
    if not os.path.exists("Vid/{}/".format(args.env)):
        os.makedirs("Vid/{}/".format(args.env))

    for z in range(args.skill_nums):
        video_writer = cv2.VideoWriter(f"Vid/skill{z}" + ".avi", fourcc, 50.0, (250, 250))
        s = env.reset()
        episode_reward = 0
        z_one_hot = np.zeros(args.skill_nums)
        z_one_hot[z] = 1
        for _ in range(env.spec.max_episode_steps):
            action = agent.get_action(s, z_one_hot)
            s_, r, done, _ = env.step(action)

            episode_reward += r
            if done:
                break
            s = s_
            I = env.render(mode='rgb_array')
            I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
            I = cv2.resize(I, (250, 250))
            video_writer.write(I)
        print(f"skill: {z}, episode reward:{episode_reward:.1f}")
        video_writer.release()
    env.close()
    cv2.destroyAllWindows()


def train_loop(env, args):
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
                                       device=args.device,
                                       env_name=args.env).to(args.device)

    total_interaction_steps = 0

    for ep in tqdm(range(args.total_episodes)):
        o = env.reset()
        episode_rew = 0
        episode_len = 0

        # sample the skill, as the skill dist is uniform to ensure max entropy
        # make the skill one-hot
        z_one_hot, z = get_one_hot_encode_skill(args.skill_nums)
        logq_zs = []

        for i in range(args.max_episode_length):
            a = agent.get_action(o, z_one_hot)
            o2, r, d, _ = env.step(a)  # the reward here is not going to be used

            episode_rew += r  # record the task rew
            episode_len += 1
            total_interaction_steps += 1

            buffer.store(o, a, r, o2, d, z_one_hot)
            o = o2
            if d:
                break

            if buffer.current_size >= args.batch_size:
                data = buffer.sample_batch(batch_size=args.batch_size)

                # get the reward given in DIAYN
                rewards = discriminator.get_score(data)

                # replace the rewards part in data
                data['rew'] = rewards
                loss_pi, loss_q1, loss_q2 = agent.update(data)

                # update the discriminator
                d_loss = discriminator.update(data)
                logq_zs.append(-d_loss)

                # wandb.log({'discriminator loss': d_loss,
                #            'actor loss': loss_pi,
                #            'q1 loss': loss_q1,
                #            'q2 loss': loss_q2}, step=total_interaction_steps)
        if len(logq_zs) > 0:
            logq_zs = sum(logq_zs) / len(logq_zs)

        # display training results
        if ep % args.display_episode_interval == 0:
            print('time:{} | episode:{}, sampled skill:{}/{}, episode_reward:{}, episode length:{}'.format(
                datetime.datetime.now().strftime("%H:%M:%S"),
                ep,
                z, args.skill_nums,
                episode_rew,
                episode_len
            ))

    agent.save_model()
    discriminator.save_model()


if __name__ == "__main__":
    import argparse
    import gym
    from goal_env.mujoco import *
    from wrapper import WrapperDictEnv

    parser = argparse.ArgumentParser()

    # env parameters
    # parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--env', type=str, default='AntEmpty-v0')

    # agent parameters
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.1)
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
    parser.add_argument('--seed', '-s', type=int, default=100)
    parser.add_argument('--total-episodes', type=int, default=3000)
    parser.add_argument('--initialize-buffer-steps', type=int, default=10000)
    parser.add_argument('--max-episode-length', type=int, default=1000)
    parser.add_argument('--update-cycles', type=int, default=1000)
    parser.add_argument('--evaluate-interval-epochs', type=int, default=5)
    parser.add_argument('--evaluation-nums', type=int, default=5)
    parser.add_argument('--display-episode-interval', type=int, default=100)
    args = parser.parse_args()

    ##########################################################################
    env = gym.make(args.env)
    env = WrapperDictEnv(env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)  # to ensure during the early random exploration the data the same

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    #########################################################################
    train_loop(env, args)
    # play(env, args)
