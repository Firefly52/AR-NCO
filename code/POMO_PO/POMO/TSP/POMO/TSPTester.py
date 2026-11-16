import importlib

import torch
import json

import os
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as DefaultModel

from utils.utils import *


class TSPTester:
    def __init__(self,
                  env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
        else:
            device = torch.device('cpu')
        self.device = device

        # ENV and MODEL
        self.env = Env(**{**self.env_params, 'device': self.device})
        model_cls = self._resolve_model_class()
        self.model = model_cls(**self.model_params).to(self.device)

        self.beta = tester_params['beta']

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = model_load.get('checkpoint_path')
        if checkpoint_fullname is None:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

        # Optional GAP evaluation against reference lengths.
        self.gap_eval_params = self.tester_params.get('gap_eval', {'enable': False})
        self.gap_enable = self.gap_eval_params.get('enable', False)
        self.ref_lengths = None
        if self.gap_enable:
            ref_path = self._resolve_ref_length_path()
            self.ref_lengths = self._load_ref_lengths(ref_path)

            required = self.tester_params['test_episodes']
            if self.ref_lengths.size(0) < required:
                raise ValueError('Reference length count ({}) is smaller than test_episodes ({})'
                                 .format(self.ref_lengths.size(0), required))

    def _resolve_ref_length_path(self):
        auto_select = self.gap_eval_params.get('auto_select_by_problem_size', True)
        explicit_path = self.gap_eval_params.get('ref_length_path')

        candidate_paths = []
        if auto_select:
            problem_size = self.env_params.get('problem_size')
            if problem_size is not None:
                candidate_paths.extend([
                    f'../../TSP/lkh_workspace/ref_lengths_lkh_size_{problem_size}.pt',
                    f'../../TSP/lkh_workspace/ref_lengths_lkh_size_{problem_size}.json',
                ])
            candidate_paths.extend([
                '../../TSP/lkh_workspace/ref_lengths_lkh.pt',
                '../../TSP/lkh_workspace/ref_lengths_lkh.json',
            ])

        if explicit_path:
            candidate_paths.append(explicit_path)

        for path in candidate_paths:
            resolved = path
            if not os.path.isabs(resolved):
                resolved = os.path.normpath(os.path.join(os.path.dirname(__file__), resolved))
            if os.path.exists(resolved):
                self.logger.info('Using GAP reference file: {}'.format(resolved))
                return resolved

        if explicit_path:
            return explicit_path
        raise ValueError('gap_eval.enable is True but no reference length file was found.')

    def _resolve_model_class(self):
        """Resolve which TSP model class to use.

        Backward compatible:
        - Default: uses TSPModel.TSPModel
        - If tester_params['model_module'] is set, imports that module and uses its TSPModel.
        """
        module_name = self.tester_params.get('model_module')
        if not module_name:
            return DefaultModel

        mod = importlib.import_module(module_name)
        if not hasattr(mod, 'TSPModel'):
            raise AttributeError("Model module '{}' must define symbol TSPModel".format(module_name))
        return getattr(mod, 'TSPModel')

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        gap_AM = AverageMeter()
        aug_gap_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score, no_aug_scores, aug_scores = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            if self.gap_enable:
                ref_batch = self.ref_lengths[episode:episode + batch_size]
                no_aug_gap = ((no_aug_scores - ref_batch) / ref_batch) * 100.0
                aug_gap = ((aug_scores - ref_batch) / ref_batch) * 100.0
                gap_AM.update(no_aug_gap.mean().item(), batch_size)
                aug_gap_AM.update(aug_gap.mean().item(), batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            if self.gap_enable:
                self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}, gap:{:.3f}%, aug_gap:{:.3f}%".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score,
                    gap_AM.avg, aug_gap_AM.avg))
            else:
                self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
                if self.gap_enable:
                    self.logger.info(" NO-AUG GAP: {:.4f}% ".format(gap_AM.avg))
                    self.logger.info(" AUGMENTATION GAP: {:.4f}% ".format(aug_gap_AM.avg))

    @torch.no_grad()
    def _test_one_batch(self, batch_size):
        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        prob_list = torch.zeros(size=(batch_size * aug_factor, self.env.pomo_size, 0), device=self.device)
        
        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor=aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_scores = -max_pomo_reward[0, :].float()  # shape: (batch,)
        no_aug_score = no_aug_scores.mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_scores = -max_aug_pomo_reward.float()  # shape: (batch,)
        aug_score = aug_scores.mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item(), no_aug_scores.cpu(), aug_scores.cpu()

    def _load_ref_lengths(self, ref_path):
        path = ref_path
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(os.path.dirname(__file__), path))

        ext = os.path.splitext(path)[1].lower()
        if ext == '.pt':
            data = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                if 'ref_lengths' in data:
                    data = data['ref_lengths']
                elif 'lengths' in data:
                    data = data['lengths']
                else:
                    raise ValueError('Unsupported .pt dict format: expected key ref_lengths or lengths')
            return torch.as_tensor(data, dtype=torch.float)

        if ext == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                if 'ref_lengths' in data:
                    data = data['ref_lengths']
                elif 'lengths' in data:
                    data = data['lengths']
                else:
                    raise ValueError('Unsupported .json dict format: expected key ref_lengths or lengths')
            return torch.as_tensor(data, dtype=torch.float)

        values = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                values.append(float(s))
        return torch.as_tensor(values, dtype=torch.float)