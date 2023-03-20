import torch
import numpy as np
from models import SkillConditionActor, MLPQFunction
from copy import deepcopy
import os


class SACAgent:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.device = self.args.device
        self.save_path = './sac_DIAYN/{}'.format(args.env)

        self.act_limit = self.env.action_space.high[0]

        # define the actor critic
        self.q1 = MLPQFunction(obs_dim=self.env.observation_space.shape[0],
                               act_dim=self.env.action_space.shape[0],
                               skill_nums=self.args.skill_nums,
                               hidden_sizes=self.args.hidden_size).to(self.device)

        self.q2 = MLPQFunction(obs_dim=self.env.observation_space.shape[0],
                               act_dim=self.env.action_space.shape[0],
                               skill_nums=self.args.skill_nums,
                               hidden_sizes=self.args.hidden_size).to(self.device)

        self.q1_targ = deepcopy(self.q1).to(self.device)
        self.q2_targ = deepcopy(self.q2).to(self.device)

        self.actor = SkillConditionActor(obs_dim=self.env.observation_space.shape[0],
                                         skill_dim=self.args.skill_nums,
                                         act_dim=self.env.action_space.shape[0],
                                         hidden_sizes=self.args.hidden_size,
                                         act_limit=self.act_limit).to(self.device)

        # define the optimizer for them
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=self.args.q_lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=self.args.q_lr)
        # the optimizer for the policy network
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.p_lr)

    def _compute_q_loss(self, o, a, r, o2, d, z_one_hot):
        q1 = self.q1(o, z_one_hot, a)
        q2 = self.q2(o, z_one_hot, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor(torch.concat([o2, z_one_hot], dim=-1))

            # Target Q-values
            q1_pi_targ = self.q1_targ(o2, z_one_hot, a2)
            q2_pi_targ = self.q2_targ(o2, z_one_hot, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = self.args.reward_scale * r + self.args.gamma * (1 - d) * (
                    q_pi_targ - self.args.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        return loss_q1, loss_q2

    def _compute_pi_loss(self, o, z):
        a_new, a_new_logprobs = self.actor(torch.concat([o, z], dim=-1))

        q1_pi = self.q1(o, z, a_new)
        q2_pi = self.q2(o, z, a_new)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (self.args.alpha * a_new_logprobs - q_pi).mean()
        return loss_pi

    def update(self, data):
        o, a, r, o2, d, z_one_hot = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data['skills']
        loss_pi = self._compute_pi_loss(o, z_one_hot)
        loss_q1, loss_q2 = self._compute_q_loss(o, a, r, o2, d, z_one_hot)

        # update the networks
        self.actor_optim.zero_grad()
        loss_pi.backward()
        self.actor_optim.step()

        self.q1_optim.zero_grad()
        loss_q1.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        loss_q2.backward()
        self.q2_optim.step()

        # update the target networks
        self._soft_update_target_networks(self.q1_targ, self.q1)
        self._soft_update_target_networks(self.q2_targ, self.q2)

        return loss_pi.item(), loss_q1.item(), loss_q2.item()

    def _soft_update_target_networks(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.args.polyak * param.data + (1 - self.args.polyak) * target_param.data)

    def get_action(self, o, skill, deterministic=False):
        obs_skill = np.concatenate([o, skill], axis=-1)
        obs_skill = torch.as_tensor(obs_skill, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            a, _ = self.actor(obs_skill, deterministic, False)
        return a.cpu().numpy()

    def save_model(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save({'actor': self.actor.state_dict(),
                    'q1': self.q1.state_dict(),
                    'q2': self.q2.state_dict()}, self.save_path + 'agent_model.pt')

    def load_model(self, path=None):
        if path == None:
            path = self.save_path + 'agent_model.pt'

        try:
            check_point = torch.load(path)
            self.actor.load_state_dict(check_point['actor'])
            self.q1.load_state_dict(check_point['q1'])
            self.q2.load_state_dict(check_point['q2'])
        except:
            print('specify a model load path')
