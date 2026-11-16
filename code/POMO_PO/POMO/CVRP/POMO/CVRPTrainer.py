
import importlib
import os

import torch
import torch.nn.functional as F
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRP.POMO.CVRPModel import CVRPModel as DefaultModel

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *


class CVRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,
                 search_params=None):

        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.search_params = search_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        self.loss_type = trainer_params['loss_type']
        self.alpha = trainer_params['alpha']

        use_cuda = self.trainer_params['use_cuda']
        if use_cuda:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
        else:
            device = torch.device('cpu')
        self.device = device

        model_cls = self._resolve_model_class()
        self.model = model_cls(**self.model_params).to(self.device)
        self.env = Env(**{**self.env_params, 'device': self.device})
        self.local_search = self._build_local_search_or_none()

        self.epoch_test_params = self.trainer_params.get('epoch_test', {
            'enable': False,
            'validation_set_path': '../../data/validation/cvrp/uniform_size_100_seed_12345_cvrp.pt',
            'episodes': 10000,
            'batch_size': 500,
        })
        self.val_env = None
        if self.epoch_test_params.get('enable', False):
            val_path = self.epoch_test_params.get(
                'validation_set_path',
                '../../data/validation/cvrp/uniform_size_100_seed_12345_cvrp.pt',
            )
            self.val_env = Env(**{**self.env_params, 'device': self.device})
            self.val_env.use_saved_problems(val_path, self.device)

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        self.start_epoch = 1
        model_load = trainer_params.get('model_load', {'enable': False})
        if model_load.get('enable', False):
            checkpoint_fullname = self._resolve_checkpoint_path(model_load)
            if not os.path.exists(checkpoint_fullname):
                raise FileNotFoundError('Checkpoint not found: {}'.format(checkpoint_fullname))

            checkpoint = torch.load(checkpoint_fullname, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            loaded_epoch = checkpoint.get('epoch', model_load.get('epoch', 0))
            self.start_epoch = 1 + loaded_epoch
            if 'result_log' in checkpoint:
                self.result_log.set_raw_data(checkpoint['result_log'])

            if model_load.get('load_optimizer', True):
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                else:
                    self.scheduler.last_epoch = loaded_epoch - 1

            self.logger.info('Saved Model Loaded from {} (epoch {})'.format(checkpoint_fullname, loaded_epoch))

        self.time_estimator = TimeEstimator()

    def _resolve_model_class(self):
        module_name = self.trainer_params.get('model_module')
        if not module_name:
            return DefaultModel

        mod = importlib.import_module(module_name)
        if not hasattr(mod, 'CVRPModel'):
            raise AttributeError("Model module '{}' must define symbol CVRPModel".format(module_name))
        return getattr(mod, 'CVRPModel')

    @staticmethod
    def _resolve_checkpoint_path(model_load):
        checkpoint_path = model_load.get('checkpoint_path') or model_load.get('model_path')
        if checkpoint_path is not None:
            return checkpoint_path
        if 'path' in model_load and 'epoch' in model_load:
            return '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        raise KeyError("model_load must set 'checkpoint_path' or 'model_path' when enable=True")

    def _build_local_search_or_none(self):
        if not self.search_params:
            return None
        try:
            from CVRPLocalSearch import CVRPLocalSearch as LocalSearch
        except ImportError:
            self.logger.warning('CVRPLocalSearch is unavailable. local_search-dependent loss will fail if selected.')
            return None
        return LocalSearch(**self.search_params)

    def _save_checkpoint(self, epoch, filename):
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'result_log': self.result_log.get_raw_data(),
        }
        torch.save(checkpoint_dict, '{}/{}'.format(self.result_folder, filename))

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=================================================================')

            self.scheduler.step()

            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            if self.val_env is not None:
                val_score = self._validate_one_epoch()
                self.result_log.append('val_score', epoch, val_score)
                self.logger.info('Epoch {:3d}: Validation Score: {:.4f}'.format(epoch, val_score))

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:
                self.logger.info('Saving log_image')
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_1'],
                    self.result_log,
                    labels=['train_score'],
                )
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_2'],
                    self.result_log,
                    labels=['train_loss'],
                )

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info('Saving trained_model')
                self._save_checkpoint(epoch, 'checkpoint-{}.pt'.format(epoch))

            self._save_checkpoint(epoch, 'last.pt')

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_1'],
                    self.result_log,
                    labels=['train_score'],
                )
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_2'],
                    self.result_log,
                    labels=['train_loss'],
                )

            if all_done:
                self.logger.info(' *** Training Done *** ')
                self.logger.info('Now, printing log array...')
                util_print_log_array(self.logger, self.result_log)

    @torch.no_grad()
    def _validate_one_epoch(self):
        self.model.eval()
        prev_eval_type = self.model.model_params.get('eval_type', 'argmax')
        self.model.model_params['eval_type'] = 'argmax'

        val_episodes = self.epoch_test_params.get('episodes', 10000)
        val_batch_size = self.epoch_test_params.get('batch_size', 500)
        if self.val_env.saved_depot_xy is not None:
            val_episodes = min(val_episodes, self.val_env.saved_depot_xy.size(0))

        self.val_env.saved_index = 0
        score_am = AverageMeter()

        try:
            episode = 0
            while episode < val_episodes:
                remaining = val_episodes - episode
                batch = min(val_batch_size, remaining)

                self.val_env.load_problems(batch, aug_factor=1)
                reset_state, _, _ = self.val_env.reset()
                self.model.pre_forward(reset_state)

                state, reward, done = self.val_env.pre_step()
                while not done:
                    selected, _ = self.model(state)
                    state, reward, done = self.val_env.step(selected)

                max_pomo_reward, _ = reward.max(dim=1)
                score = -max_pomo_reward.float().mean().item()
                score_am.update(score, batch)

                episode += batch

            return score_am.avg
        finally:
            self.model.model_params['eval_type'] = prev_eval_type

    def _train_one_epoch(self, epoch):
        score_am = AverageMeter()
        loss_am = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        sample_log_interval = self.trainer_params.get('sample_log_interval', 3200)
        episode = 0
        next_log_sample = sample_log_interval
        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_am.update(avg_score, batch_size)
            loss_am.update(avg_loss, batch_size)

            episode += batch_size

            while episode >= next_log_sample and next_log_sample <= train_num_episode:
                self.logger.info('Epoch {:3d}: Train {:5d}/{:5d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                 .format(epoch, episode, train_num_episode, 100.0 * episode / train_num_episode,
                                         score_am.avg, loss_am.avg))
                next_log_sample += sample_log_interval

        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100.0 * episode / train_num_episode, score_am.avg, loss_am.avg))

        return score_am.avg, loss_am.avg

    def _train_one_batch(self, batch_size):
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0), device=self.device)

        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        if self.loss_type == 'po_loss':
            loss = self.preference_among_pomo_loss_fn(reward, prob_list)
        elif self.loss_type == 'rl_loss':
            loss = self.rl_loss_fn(reward, prob_list)
        else:
            raise NotImplementedError

        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return score_mean.item(), loss.item()

    def rl_loss_fn(self, reward, prob_list):
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        log_prob = prob_list.log().sum(dim=2)
        loss = -advantage * log_prob
        return loss.mean()

    def preference_among_pomo_loss_fn(self, reward, prob):
        preference = reward[:, :, None] > reward[:, None, :]

        log_prob = self.alpha * torch.log(prob).sum(2)
        log_prob_pair = log_prob[:, :, None] - log_prob[:, None, :]
        pf_log = torch.log(F.sigmoid(log_prob_pair))
        loss = -torch.mean(pf_log * preference)

        return loss
