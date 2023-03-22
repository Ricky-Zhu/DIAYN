import torch
import os
import torch.nn as nn


class SkillDiscriminator(nn.Module):
    def __init__(self, obs_dim, skill_nums, hidden_size, lr, device, env_name):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, skill_nums))
        self.static_log_skill_prob = torch.log(
            torch.as_tensor(1 / skill_nums + 1e-6, dtype=torch.float32, device=device))
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.skill_nums = skill_nums
        self.optim = torch.optim.Adam(params=self.net.parameters(), lr=lr)

        self.save_path = './sac_DIAYN/{}/'.format(env_name)

    def forward(self, obs):
        skill_logits = self.net(obs)
        return skill_logits

    def get_score(self, data):
        o2 = data['obs2']
        z_one_hot = data['skills']
        pred_skills_logits = self(o2)  # get the predict logits of the skills
        targ_skill_probs = (self.softmax(pred_skills_logits.clone().detach()) * z_one_hot).sum(-1)
        score = torch.log(targ_skill_probs) - self.static_log_skill_prob
        return score

    def update(self, data):
        o2 = data['obs']
        z = data['skills']
        pred_skills = self(o2)
        loss = self.loss(pred_skills, z)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def save_model(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.state_dict(), self.save_path + 'discriminator_model.pt')

    def load_model(self, path=None):
        if path == None:
            path = self.save_path + 'discriminator_model.pt'
        self.load_state_dict(torch.load(path))
