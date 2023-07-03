from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.sac_utils import TanhTransform
from cs285.infrastructure.sac_utils import soft_update_params
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
import torch
from torch import nn

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: 
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic  

        #TRY THE ABOVE LINE AS A FIX IF NEED BE


        #FIX ALL THE TORCH NP STUFF BEFORE RUNNING THE CODE

        #we can try the mean (argmax) if we want to later

        ob_no = ptu.from_numpy(ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)
        # ac_na = TanhTransform()(ptu.from_numpy(ac_na))
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        next_ob_dist = self.actor(next_ob_no)
        next_ac = next_ob_dist.rsample()
        # next_ac = ptu.from_numpy(next_ac)
        

        q1, q2 = self.critic_target(next_ob_no, next_ac)
        q1, q2 = q1.squeeze(), q2.squeeze()

        target_value = re_n + self.gamma * (1 - terminal_n) * (torch.minimum(q1, q2) - self.actor.alpha * next_ob_dist.log_prob(next_ac).sum(1))
        current_q_1, current_q_2 = self.critic(ob_no, ac_na)
        current_q_1, current_q_2 = current_q_1.squeeze(), current_q_2.squeeze()
        target_value = target_value.detach()
        critic_loss = self.critic.loss(current_q_1, target_value) + self.critic.loss(current_q_2, target_value)
        
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        return critic_loss.item()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO 
        # 1. Implement the following pseudocode:

        critic_loss = 0

        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss += self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
        if self.training_step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target, self.critic_tau)
            # print("critic_loss ", critic_loss)
        
        actor_loss, alpha_loss, temperature = 0, 0, self.actor.init_temperature
        if self.training_step % self.actor_update_frequency == 0:
            for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
                actor_l, alpha_l, alpha = self.actor.update(ob_no, self.critic)
                actor_loss += actor_l
                alpha_loss += alpha_l
                temperature = alpha
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)

        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss / self.agent_params['num_critic_updates_per_agent_update']
        loss['Actor_Loss'] = actor_loss / self.agent_params['num_actor_updates_per_agent_update']
        loss['Alpha_Loss'] = alpha_loss / self.agent_params['num_actor_updates_per_agent_update']
        loss['Temperature'] = temperature
        self.training_step += 1
        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
