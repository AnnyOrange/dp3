import sys
import os
import wandb
import numpy as np
import torch
import collections
import tqdm
from diffusion_policy_3d.env import MetaWorldEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint
import json
import imageio
class MetaworldRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        print("device---------------------",device)

        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        # self.eval_episodes = eval_episodes
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        self.output_dir = output_dir

    def run(self, policy: BasePolicy, save_video=False):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env
        # print(self.n_action_steps)
         
        # t = np.zeros(n_envs)
        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            t = 0
            n_envs = 1
            state_dim = 4
            self.temporal_agg = False
            num_samples = 10
            if self.temporal_agg:
                    all_time_actions = torch.zeros(
                        [self.n_action_steps, self.n_action_steps-1, n_envs, state_dim]
                    ).to(device)   
                    all_time_samples = torch.zeros(
                        [self.n_action_steps, self.n_action_steps-1, n_envs, num_samples, state_dim-1]
                    ).to(device) 
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            num_samples = 10
            step = 0
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)
                    # import pdb;pdb.set_trace()
                    sample_dict = policy.get_samples(obs_dict_input, num_samples=num_samples)
                    
                
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                np_sample_dict = dict_apply(sample_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                # print(np_action_dict['action'].shape)
                action = np_action_dict['action'].squeeze(0)
                
                sample = np_sample_dict['action']
                sample = sample.reshape(num_samples,sample.shape[0]//num_samples,sample.shape[1],sample.shape[2])
                # print("sample",sample.shape) # (10, 1, 3, 4)
                # print("action",action.shape) # (1,3,4)
                env_action = np_action_dict['action']
                # all_time_actions_i = all_time_actions[:,:,episode_idx,:].unsqueeze(2)
                # all_time_samples_i = all_time_samples[:,:,i,:,:].unsqueeze(2)
                # import pdb;pdb.set_trace()
                closeloop = False
                if closeloop:
                    if self.temporal_agg:
                        all_actions = torch.from_numpy(env_action).float().to(device)
                        all_samples = torch.from_numpy(sample).float().to(device)
                        all_samples = all_samples.permute(2,1,0,3)  # (16,28,10,7) #(3,1,10,4)
                        # all_actions扩维度 最开始增加维度
                        all_actions = all_actions.permute(1,0,2) # (16,28,7) #(3,1,4)
                        # print(all_actions.shape)
                        all_time_actions[[-1], :self.n_action_steps-1] = all_actions  
                        # print(all_time_actions.shape) #(4,3,20,4)
                        actions_for_curr_step = all_time_actions[:, 0]  
                        
                        actions_populated = torch.all(actions_for_curr_step[:,:,0] != 0, axis=-1)  
                        all_time_samples[[-1],  :self.n_action_steps-1] = all_samples[:,:,:,:3] 
                        samples_for_curr_step = all_time_samples[:, 0]  
                        
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        samples_for_curr_step = samples_for_curr_step[actions_populated]
                        # print(samples_for_curr_step.shape) # (1,1,10,4)  1 10 20 4     10 20 3 4 1  10 20 1 4
                        # import pdb;pdb.set_trace()
                        
                        entropy = torch.mean(torch.var(samples_for_curr_step.permute(0,2,1,3).flatten(0,1),dim=0,keepdim=True),dim=-1,keepdim=True)
                        entropy = entropy.permute(1,0,2).detach().cpu().numpy()

                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                                torch.from_numpy(exp_weights).to(device).unsqueeze(dim=1).unsqueeze(dim=-1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True).permute(1,0,2)
                        action = raw_action.detach().cpu().numpy()

                        # move temporal ensemble window
                        all_time_actions_temp = torch.zeros_like(all_time_actions)
                        all_time_actions_temp[:-1,:-1] = all_time_actions[1:,1:]
                        all_time_actions = all_time_actions_temp
                        del all_time_actions_temp
                        all_time_samples_temp = torch.zeros_like(all_time_samples)
                        all_time_samples_temp[:-1,:-1] = all_time_samples[1:,1:]
                        all_time_samples = all_time_samples_temp
                        del all_time_samples_temp
                        
                        t+=1
                    else:
                        sample = np.expand_dims(sample[:,:,0,:], axis=2).transpose(1, 0, 2, 3)
                        # print(sample.shape)
                        entropy = torch.mean(
                            torch.var(
                                torch.from_numpy(sample).float().to(device).flatten(0, 1),
                                dim=0,
                                keepdim=True
                            ),
                            dim=-1,
                            keepdim=True
                        ).detach().cpu().numpy()
                        if action.ndim == 2 and action.shape[0]!=1:
                            action = np.expand_dims(action[0,:],axis=0)
                            # print(action.shape)
                        # print(entropy.shape)
                        # entropy.reshape(action.shape)
                else:
                    sample = np.expand_dims(sample[:,:,0,:], axis=2).transpose(1, 0, 2, 3)
                        # print(sample.shape)
                    entropy = torch.mean(
                        torch.var(
                            torch.from_numpy(sample).float().to(device).flatten(0, 1),
                            dim=0,
                            keepdim=True
                        ),
                        dim=-1,
                        keepdim=True
                    ).detach().cpu().numpy()
                    entropy = np.tile(entropy, (action.shape[0], 1))
                if action.ndim == 3:
                    action = action.squeeze(0)
                if entropy.ndim == 3:
                    entropy = entropy.squeeze(0)
                
                action = np.concatenate((action, entropy),axis=-1)
                # print(action.shape)
                # import pdb;pdb.set_trace()
                action = action[2, :][np.newaxis, :]

                obs, reward, done, info = env.step(action)
                
                traj_reward += reward
                done = np.all(done)
                step += int(not done)
                is_success = is_success or max(info['success'])
            # print("step",step)
            # print(len(env.statelist))
            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            
        avg_steps = len(env.statelist)/self.eval_episodes
        
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        videos = env.env.get_video()
        video_path = os.path.join(self.output_dir, 'eval_videos', f'episode_xyz_{episode_idx}.mp4')  # 保存路径
        os.makedirs(os.path.dirname(video_path), exist_ok=True)  
        self.save_video_to_file(videos, save_path = video_path, fps = self.fps)
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        if save_video:
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()
        videos = None

        return log_data
    def save_video_to_file(self,videos, save_path, fps=10):
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # 如果 shape 是 5 维，去掉 batch 维度
        # print(videos.shape)
        videos = np.transpose(videos, (0, 2, 3, 1)) 
        with imageio.get_writer(save_path, fps=fps) as video_writer:
            for frame in videos:
                video_writer.append_data(frame)