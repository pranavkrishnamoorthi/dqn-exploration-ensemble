# from ctypes.wintypes import HACCEL
import re
from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools
from torch import distributions


class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim
        

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        
        observation = ptu.from_numpy(observation.astype(np.float32))
        action_dist = self(observation)
        if sample:
            return ptu.to_numpy(action_dist.rsample())

        # this command might not be right...check if need to when debugging
        return ptu.to_numpy(action_dist.mean)



    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            scale = torch.exp(self.logstd.clip(min = self.log_std_bounds[0], max = self.log_std_bounds[1]))
            action_distribution = sac_utils.SquashedNormal(batch_mean, scale)
            # scale_tril = torch.diag(torch.exp(self.logstd))
            # batch_dim = batch_mean.shape[0]
            # batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)

            # action_distribution = distributions.MultivariateNormal(
            #     batch_mean,
            #     scale_tril=batch_scale_tril,
            # )
            return action_distribution

        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        
        # we can have this return a log of the distribution
        # return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha valu
        
        obs = ptu.from_numpy(obs)
        policy_action_dist = self(obs)
        # print("actions before tanh squash ", actions[:10])
        # print("actions", actions[:10])
        actions = policy_action_dist.rsample()
        q1, q2 = critic(obs, actions)
        actor_loss =  (self.alpha * policy_action_dist.log_prob(actions).sum(1) - torch.minimum(q1, q2)).mean()
        
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()   

        # maybe I need to reattach self.alpha?
        new_policy_action_dist = self(obs)
        actions = new_policy_action_dist.sample()
        log_probs = new_policy_action_dist.log_prob(actions)
        log_probs = log_probs.detach()

        
        alpha_loss = -self.alpha * (log_probs.mean() + self.target_entropy)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        # print('actor loss', actor_loss.item())
        # print('alpha loss', alpha_loss.item())

        return actor_loss.item(), alpha_loss.item(), self.alpha