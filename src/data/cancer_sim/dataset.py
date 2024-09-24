import numpy as np
from torch.utils.data import Dataset
import logging
import inspect
from copy import deepcopy
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.dataset_collection import SyntheticDatasetCollection
from src.data.cancer_sim.cancer_simulation import TUMOUR_DEATH_THRESHOLD
from src.data.cancer_sim.cancer_simulation import generate_params, get_scaling_params, simulate_factual, \
    simulate_counterfactual_1_step, simulate_counterfactuals_treatment_seq


logger = logging.getLogger(__name__)


class SyntheticCancerDataset(Dataset):
    """
    Pytorch-style Dataset of Tumor Growth Simulator datasets
    """

    def __init__(self,
                 chemo_coeff: float,
                 radio_coeff: float,
                 num_patients: int,
                 window_size: int,
                 seq_length: int,
                 subset_name: str,
                 mode='factual',
                 projection_horizon: int = None,
                 seed=None,
                 lag: int = 0,
                 cf_seq_mode: str = 'sliding_treatment',
                 treatment_mode: str = 'multiclass',
                 latent_confounder: bool = None,
                 u_ratio: float = 1,
                 chemo_iv_ratio: float = 1,
                 radio_iv_ratio: float = 1):
        """
        Args:
            chemo_coeff: Confounding coefficient of chemotherapy
            radio_coeff: Confounding coefficient of radiotherapy
            num_patients: Number of patients in dataset
            window_size: Used for biased treatment assignment
            seq_length: Max length of time series
            subset_name: train / val / test
            mode: factual / counterfactual_one_step / counterfactual_treatment_seq
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            seed: Random seed
            lag: Lag for treatment assignment window
            cf_seq_mode: sliding_treatment / random_trajectories
            treatment_mode: multiclass / multilabel
        """

        if seed is not None:
            np.random.seed(seed)

        self.chemo_coeff = chemo_coeff
        self.radio_coeff = radio_coeff
        self.window_size = window_size
        self.num_patients = num_patients
        self.latent_confounder = latent_confounder
        self.params = generate_params(num_patients, chemo_coeff=chemo_coeff, radio_coeff=radio_coeff,
                                      window_size=window_size, lag=lag)
        self.subset_name = subset_name

        if self.latent_confounder:
            if mode == 'factual':
                self.data = simulate_factual(self.params, seq_length, u_ratio, chemo_iv_ratio, radio_iv_ratio, with_latent_confounder=True)
                #logger.info(f'0000self.data.keys: {self.data.keys()}')
            elif mode == 'counterfactual_one_step':
                self.data = simulate_counterfactual_1_step(self.params, seq_length, u_ratio, chemo_iv_ratio, radio_iv_ratio, with_latent_confounder=True)
                #logger.info(f'11111self.data.keys: {self.data.keys()}')
            elif mode == 'counterfactual_treatment_seq':
                assert projection_horizon is not None
                self.data = simulate_counterfactuals_treatment_seq(self.params, seq_length, projection_horizon, u_ratio, chemo_iv_ratio, radio_iv_ratio, cf_seq_mode, with_latent_confounder=True)
                #logger.info(f'99999self.data.keys: {self.data.keys()}')
        else:
            if mode == 'factual':
                self.data = simulate_factual(self.params, seq_length, u_ratio, chemo_iv_ratio, radio_iv_ratio)
            elif mode == 'counterfactual_one_step':
                self.data = simulate_counterfactual_1_step(self.params, seq_length, u_ratio, chemo_iv_ratio, radio_iv_ratio)
            elif mode == 'counterfactual_treatment_seq':
                assert projection_horizon is not None
                self.data = simulate_counterfactuals_treatment_seq(self.params, seq_length, projection_horizon, u_ratio, chemo_iv_ratio, radio_iv_ratio, cf_seq_mode)

        self.processed = False
        self.processed_sequential = False
        self.processed_autoregressive = False
        self.treatment_mode = treatment_mode
        self.exploded = False

        self.norm_const = TUMOUR_DEATH_THRESHOLD

    def __getitem__(self, index) -> dict:
        result = {k: v[index] for k, v in self.data.items() if hasattr(v, '__len__') and len(v) == len(self)}
        #logger.info(f'getitem data keys: {self.data.keys()}')
        #logger.info(f'dataset length: {len(self)}')
        '''for k,v in self.data.items():
            logger.info(f'key: {k}, length: {len(v)}')
        logger.info(f'subset_name: {self.subset_name},  result content keys1: {result.keys()}')'''
        if hasattr(self, 'encoder_r'):
            if 'original_index' in self.data:
                result.update({'encoder_r': self.encoder_r[int(result['original_index'])]})
            else:
                result.update({'encoder_r': self.encoder_r[index]})
        return result

    def __len__(self):
        return self.data['current_covariates'].shape[0]

    def get_scaling_params(self):
        return get_scaling_params(self.data)

    def process_data(self, scaling_params):
        """
        Pre-process dataset for one-step-ahead prediction
        Args:
            scaling_params: dict of standard normalization parameters (calculated with train subset)
        """
        if not self.processed:
            logger.info(f'Processing {self.subset_name} dataset before training')

            mean, std = scaling_params
            logger.info(f'mean shape: {mean.shape}, std:shape {std.shape}')

            horizon = 1
            offset = 1

            mean['chemo_application'] = 0
            mean['radio_application'] = 0
            std['chemo_application'] = 1
            std['radio_application'] = 1

            input_means = mean[['cancer_volume', 'patient_types', 'chemo_application', 'radio_application', 'U', 'chemo_iv', 'radio_iv']].values.flatten()
            input_stds = std[['cancer_volume', 'patient_types', 'chemo_application', 'radio_application', 'U', 'chemo_iv', 'radio_iv']].values.flatten()
            logger.info(f'input_means shape: {input_means.shape}, input_stds shape: {input_stds.shape}')

            # Continuous values
            cancer_volume = (self.data['cancer_volume'] - mean['cancer_volume']) / std['cancer_volume']
            logger.info(f'cancer_volume: {cancer_volume.shape}')
            patient_types = (self.data['patient_types'] - mean['patient_types']) / std['patient_types']
            logger.info(f'patient_types: {patient_types.shape}')
            patient_types = np.stack([patient_types for t in range(cancer_volume.shape[1])], axis=1)

            # Binary application
            chemo_application = self.data['chemo_application']
            logger.info(f'chemo_application shape: {chemo_application.shape}')
            radio_application = self.data['radio_application']
            logger.info(f'radio_application shape: {radio_application.shape}')
            sequence_lengths = self.data['sequence_lengths']
            logger.info(f'sequence_lengths shape: {sequence_lengths.shape}')
            # Convert prev_treatments to one-hot encoding

            treatments = np.concatenate(
                [chemo_application[:, :-offset, np.newaxis], radio_application[:, :-offset, np.newaxis]], axis=-1)
            logger.info(f'treatments shape111: {treatments.shape}')

            if self.treatment_mode == 'multiclass':
                one_hot_treatments = np.zeros(shape=(treatments.shape[0], treatments.shape[1], 4))
                for patient_id in range(treatments.shape[0]):
                    for timestep in range(treatments.shape[1]):
                        if (treatments[patient_id][timestep][0] == 0 and treatments[patient_id][timestep][1] == 0):
                            one_hot_treatments[patient_id][timestep] = [1, 0, 0, 0]
                        elif (treatments[patient_id][timestep][0] == 1 and treatments[patient_id][timestep][1] == 0):
                            one_hot_treatments[patient_id][timestep] = [0, 1, 0, 0]
                        elif (treatments[patient_id][timestep][0] == 0 and treatments[patient_id][timestep][1] == 1):
                            one_hot_treatments[patient_id][timestep] = [0, 0, 1, 0]
                        elif (treatments[patient_id][timestep][0] == 1 and treatments[patient_id][timestep][1] == 1):
                            one_hot_treatments[patient_id][timestep] = [0, 0, 0, 1]

                one_hot_previous_treatments = one_hot_treatments[:, :-1, :]

                self.data['prev_treatments'] = one_hot_previous_treatments
                self.data['current_treatments'] = one_hot_treatments

            elif self.treatment_mode == 'multilabel':
                self.data['prev_treatments'] = treatments[:, :-1, :]
                self.data['current_treatments'] = treatments

            logger.info(f'treatments shape222: {treatments.shape}')

            current_covariates = np.concatenate([cancer_volume[:, :-offset, np.newaxis], patient_types[:, :-offset, np.newaxis]],
                                                axis=-1)
            '''U = self.data['U']
            chemo_iv = self.data['chemo_iv']
            radio_iv = self.data['radio_iv']'''

            U = (self.data['U'] - mean['U']) / std['U']
            chemo_iv = (self.data['chemo_iv'] - mean['chemo_iv']) / std['chemo_iv']
            radio_iv = (self.data['radio_iv'] - mean['radio_iv']) / std['radio_iv']

            if self.latent_confounder:
                U = U[:,:-offset, np.newaxis]
                chemo_iv = chemo_iv[:, :-offset, np.newaxis]
                radio_iv = radio_iv[:, :-offset, np.newaxis]
            self.data['U'] = U
            self.data['chemo_iv'] = chemo_iv
            self.data['radio_iv'] = radio_iv

            outputs = cancer_volume[:, horizon:, np.newaxis]

            output_means = mean[['cancer_volume']].values.flatten()[0]  # because we only need scalars here
            output_stds = std[['cancer_volume']].values.flatten()[0]

            # Add active entires
            active_entries = np.zeros(outputs.shape)

            for i in range(sequence_lengths.shape[0]):
                sequence_length = int(sequence_lengths[i])
                active_entries[i, :sequence_length, :] = 1

            self.data['current_covariates'] = current_covariates
            self.data['outputs'] = outputs
            self.data['active_entries'] = active_entries

            self.data['unscaled_outputs'] = (outputs * std['cancer_volume'] + mean['cancer_volume'])

            self.scaling_params = {
                'input_means': input_means,
                'inputs_stds': input_stds,
                'output_means': output_means,
                'output_stds': output_stds
            }

            # Unified data format
            self.data['prev_outputs'] = current_covariates[:, :, :1]
            self.data['static_features'] = current_covariates[:, 0, 1:]
            zero_init_treatment = np.zeros(shape=[current_covariates.shape[0], 1, self.data['prev_treatments'].shape[-1]])
            self.data['prev_treatments'] = np.concatenate([zero_init_treatment, self.data['prev_treatments']], axis=1)

            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'func {inspect.currentframe().f_code.co_name} Shape of processed {self.subset_name} data_shapes: {data_shapes}')

            self.processed = True
        else:
            logger.info(f'{self.subset_name} Dataset already processed')

        return self.data

    def explode_trajectories(self, projection_horizon):
        assert self.processed

        logger.info(f'Exploding {self.subset_name} dataset before testing (multiple sequences)')

        outputs = self.data['outputs']
        prev_outputs = self.data['prev_outputs']
        sequence_lengths = self.data['sequence_lengths']
        # vitals = self.data['vitals']
        # next_vitals = self.data['next_vitals']
        active_entries = self.data['active_entries']
        current_treatments = self.data['current_treatments']
        previous_treatments = self.data['prev_treatments']
        static_features = self.data['static_features']
        U = self.data['U']
        chemo_iv = self.data['chemo_iv']
        radio_iv = self.data['radio_iv']
        if 'stabilized_weights' in self.data:
            stabilized_weights = self.data['stabilized_weights']

        num_patients, max_seq_length, num_features = outputs.shape
        num_seq2seq_rows = num_patients * max_seq_length
        logger.info(f'num_patients: {num_seq2seq_rows}, max_seq_length: {max_seq_length} &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        logger.info(f'num_seq2seq_rows: {num_seq2seq_rows}&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

        seq2seq_previous_treatments = np.zeros((num_seq2seq_rows, max_seq_length, previous_treatments.shape[-1]))
        seq2seq_current_treatments = np.zeros((num_seq2seq_rows, max_seq_length, current_treatments.shape[-1]))
        seq2seq_static_features = np.zeros((num_seq2seq_rows, static_features.shape[-1]))
        seq2seq_outputs = np.zeros((num_seq2seq_rows, max_seq_length, outputs.shape[-1]))
        seq2seq_prev_outputs = np.zeros((num_seq2seq_rows, max_seq_length, prev_outputs.shape[-1]))
        # seq2seq_vitals = np.zeros((num_seq2seq_rows, max_seq_length, vitals.shape[-1]))
        # seq2seq_next_vitals = np.zeros((num_seq2seq_rows, max_seq_length - 1, next_vitals.shape[-1]))
        seq2seq_active_entries = np.zeros((num_seq2seq_rows, max_seq_length, active_entries.shape[-1]))
        seq2seq_U = np.zeros((num_seq2seq_rows, max_seq_length, U.shape[-1]))
        seq2seq_chemo_iv = np.zeros((num_seq2seq_rows, max_seq_length, chemo_iv.shape[-1]))
        seq2seq_radio_iv = np.zeros((num_seq2seq_rows, max_seq_length, radio_iv.shape[-1]))
        seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)
        if 'stabilized_weights' in self.data:
            seq2seq_stabilized_weights = np.zeros((num_seq2seq_rows, max_seq_length))

        total_seq2seq_rows = 0  # we use this to shorten any trajectories later

        '''
        举例：
        projection_horizon=3
        sequence = [3, 5, 8, 7, 2, 0, 4, 9, 1, 6]
        seq2seq_sequences =
        [[3. 5. 8. 7. 0. 0. 0. 0. 0. 0.]
         [3. 5. 8. 7. 2. 0. 0. 0. 0. 0.]
         [3. 5. 8. 7. 2. 0. 0. 0. 0. 0.]
         [3. 5. 8. 7. 2. 0. 4. 0. 0. 0.]
         [3. 5. 8. 7. 2. 0. 4. 9. 0. 0.]
         [3. 5. 8. 7. 2. 0. 4. 9. 1. 0.]
         [3. 5. 8. 7. 2. 0. 4. 9. 1. 6.]]
        seq2seq_lengths =
        [4  5  6  7  8  9 10]

        '''
        for i in range(num_patients):
            sequence_length = int(sequence_lengths[i])

            for t in range(projection_horizon, sequence_length):  # shift outputs back by 1
                seq2seq_active_entries[total_seq2seq_rows, :(t + 1), :] = active_entries[i, :(t + 1), :]
                if 'stabilized_weights' in self.data:
                    seq2seq_stabilized_weights[total_seq2seq_rows, :(t + 1)] = stabilized_weights[i, :(t + 1)]
                seq2seq_previous_treatments[total_seq2seq_rows, :(t + 1), :] = previous_treatments[i, :(t + 1), :]
                seq2seq_current_treatments[total_seq2seq_rows, :(t + 1), :] = current_treatments[i, :(t + 1), :]
                seq2seq_outputs[total_seq2seq_rows, :(t + 1), :] = outputs[i, :(t + 1), :]
                seq2seq_prev_outputs[total_seq2seq_rows, :(t + 1), :] = prev_outputs[i, :(t + 1), :]
                # seq2seq_vitals[total_seq2seq_rows, :(t + 1), :] = vitals[i, :(t + 1), :]
                # seq2seq_next_vitals[total_seq2seq_rows, :min(t + 1, sequence_length - 1), :] = \
                #     next_vitals[i, :min(t + 1, sequence_length - 1), :]
                seq2seq_U[total_seq2seq_rows, :(t + 1), :] = U[i, :(t + 1), :]
                seq2seq_chemo_iv[total_seq2seq_rows, :(t + 1), :] = chemo_iv[i, :(t + 1), :]
                seq2seq_radio_iv[total_seq2seq_rows, :(t + 1), :] = radio_iv[i, :(t + 1), :]
                seq2seq_sequence_lengths[total_seq2seq_rows] = t + 1
                seq2seq_static_features[total_seq2seq_rows] = static_features[i]

                total_seq2seq_rows += 1

        # Filter everything shorter
        seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows, :, :]
        seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]
        seq2seq_static_features = seq2seq_static_features[:total_seq2seq_rows, :]
        seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
        seq2seq_prev_outputs = seq2seq_prev_outputs[:total_seq2seq_rows, :, :]
        # seq2seq_vitals = seq2seq_vitals[:total_seq2seq_rows, :, :]
        # seq2seq_next_vitals = seq2seq_next_vitals[:total_seq2seq_rows, :, :]
        seq2seq_U = seq2seq_U[:total_seq2seq_rows, :, :]
        seq2seq_chemo_iv = seq2seq_chemo_iv[:total_seq2seq_rows, :, :]
        seq2seq_radio_iv = seq2seq_radio_iv[:total_seq2seq_rows, :, :]
        seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
        seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]
        logger.info(f'total_seq2seq_rows: {total_seq2seq_rows}&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

        if 'stabilized_weights' in self.data:
            seq2seq_stabilized_weights = seq2seq_stabilized_weights[:total_seq2seq_rows]

        new_data = {
            'prev_treatments': seq2seq_previous_treatments,
            'current_treatments': seq2seq_current_treatments,
            'static_features': seq2seq_static_features,
            'prev_outputs': seq2seq_prev_outputs,
            'outputs': seq2seq_outputs,
            # 'vitals': seq2seq_vitals,
            # 'next_vitals': seq2seq_next_vitals,
            'U': seq2seq_U,
            'chemo_iv': seq2seq_chemo_iv,
            'radio_iv': seq2seq_radio_iv,
            'unscaled_outputs': seq2seq_outputs * self.scaling_params['output_stds'] + self.scaling_params['output_means'],
            'sequence_lengths': seq2seq_sequence_lengths,
            'active_entries': seq2seq_active_entries,
        }
        if 'stabilized_weights' in self.data:
            new_data['stabilized_weights'] = seq2seq_stabilized_weights

        self.data = new_data
        self.exploded = True

        #data_shapes = {k: v.shape for k, v in self.data.items()}
        #logger.info(f'func {inspect.currentframe().f_code.co_name} Shape of processed {self.subset_name} data: {data_shapes}')

    def process_sequential(self, encoder_r, projection_horizon, save_encoder_r=False):
        """
        Pre-process dataset for multiple-step-ahead prediction: explodes dataset to a larger one with rolling origin
        Args:
            encoder_r: Representations of encoder
            projection_horizon: Projection horizon
            save_encoder_r: Save all encoder representations (for cross-attention of EDCT)
        """

        assert self.processed

        if not self.processed_sequential:
            logger.info(f'Processing {self.subset_name} dataset before training (multiple sequences)')

            outputs = self.data['outputs']
            sequence_lengths = self.data['sequence_lengths']
            active_entries = self.data['active_entries']
            current_treatments = self.data['current_treatments']
            previous_treatments = self.data['prev_treatments'][:, 1:, :]  # Without zero_init_treatment
            current_covariates = self.data['current_covariates']
            U = self.data['U']
            chemo_iv = self.data['chemo_iv']
            radio_iv = self.data['radio_iv']
            stabilized_weights = self.data['stabilized_weights'] if 'stabilized_weights' in self.data else None

            num_patients, seq_length, num_features = outputs.shape
            logger.info(f'num_patients: {num_patients}@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

            num_seq2seq_rows = num_patients * seq_length

            seq2seq_state_inits = np.zeros((num_seq2seq_rows, encoder_r.shape[-1]))
            seq2seq_active_encoder_r = np.zeros((num_seq2seq_rows, seq_length))
            seq2seq_original_index = np.zeros((num_seq2seq_rows, ))
            seq2seq_previous_treatments = np.zeros((num_seq2seq_rows, projection_horizon, previous_treatments.shape[-1]))
            seq2seq_current_treatments = np.zeros((num_seq2seq_rows, projection_horizon, current_treatments.shape[-1]))
            seq2seq_current_covariates = np.zeros((num_seq2seq_rows, projection_horizon, current_covariates.shape[-1]))
            seq2seq_outputs = np.zeros((num_seq2seq_rows, projection_horizon, outputs.shape[-1]))
            seq2seq_active_entries = np.zeros((num_seq2seq_rows, projection_horizon, active_entries.shape[-1]))
            seq2seq_U = np.zeros((num_seq2seq_rows, projection_horizon, U.shape[-1]))
            seq2seq_chemo_iv = np.zeros((num_seq2seq_rows, projection_horizon, chemo_iv.shape[-1]))
            seq2seq_radio_iv = np.zeros((num_seq2seq_rows, projection_horizon, radio_iv.shape[-1]))
            seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)
            seq2seq_stabilized_weights = np.zeros((num_seq2seq_rows, projection_horizon + 1)) \
                if stabilized_weights is not None else None
            logger.info(f'num_seq2seq_rows: {num_seq2seq_rows}@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

            total_seq2seq_rows = 0  # we use this to shorten any trajectories later

            '''假如outputs = [3, 5, 8, 7, 2, 0, 4, 9, 1, 6]
                  sequence_length = 10
               num_seq2seq_rows = 10 - projection_horizon = 7  # 生成 7 个子序列
                seq2seq_outputs = np.zeros((7, 3))  # 每个子序列长度为 3
                seq2seq_outputs[0, :3] = outputs[1:4]  # 子序列 [5, 8, 7]
                seq2seq_outputs[1, :3] = outputs[2:5]  # 子序列 [8, 7, 2]

                seq2seq_outputs = [
                  [5, 8, 7],
                  [8, 7, 2],
                  [7, 2, 0],
                  [2, 0, 4],
                  [0, 4, 9],
                  [4, 9, 1],
                  [9, 1, 6]
                ]
            '''

            for i in range(num_patients):

                sequence_length = int(sequence_lengths[i])

                for t in range(1, sequence_length - projection_horizon):  # shift outputs back by 1
                    seq2seq_state_inits[total_seq2seq_rows, :] = encoder_r[i, t - 1, :]  # previous state output
                    seq2seq_original_index[total_seq2seq_rows] = i
                    seq2seq_active_encoder_r[total_seq2seq_rows, :t] = 1.0

                    max_projection = min(projection_horizon, sequence_length - t)

                    seq2seq_active_entries[total_seq2seq_rows, :max_projection, :] = active_entries[i, t:t + max_projection, :]
                    seq2seq_previous_treatments[total_seq2seq_rows, :max_projection, :] = \
                        previous_treatments[i, t - 1:t + max_projection - 1, :]
                    seq2seq_current_treatments[total_seq2seq_rows, :max_projection, :] = \
                        current_treatments[i, t:t + max_projection, :]
                    seq2seq_outputs[total_seq2seq_rows, :max_projection, :] = outputs[i, t:t + max_projection, :]
                    seq2seq_U[total_seq2seq_rows, :max_projection, :] = U[i, t:t + max_projection, :]
                    seq2seq_chemo_iv[total_seq2seq_rows, :max_projection, :] = chemo_iv[i, t:t + max_projection, :]
                    seq2seq_radio_iv[total_seq2seq_rows, :max_projection, :] = radio_iv[i, t:t + max_projection, :]
                    seq2seq_sequence_lengths[total_seq2seq_rows] = max_projection
                    seq2seq_current_covariates[total_seq2seq_rows, :max_projection, :] = \
                        current_covariates[i, t:t + max_projection, :]

                    if seq2seq_stabilized_weights is not None:  # Also including SW of one-step-ahead prediction
                        seq2seq_stabilized_weights[total_seq2seq_rows, :] = stabilized_weights[i, t - 1:t + max_projection]

                    total_seq2seq_rows += 1

            # Filter everything shorter
            seq2seq_state_inits = seq2seq_state_inits[:total_seq2seq_rows, :]
            seq2seq_original_index = seq2seq_original_index[:total_seq2seq_rows]
            seq2seq_active_encoder_r = seq2seq_active_encoder_r[:total_seq2seq_rows, :]
            seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows, :, :]
            seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]
            seq2seq_current_covariates = seq2seq_current_covariates[:total_seq2seq_rows, :, :]
            seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
            seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
            seq2seq_U = seq2seq_U[:total_seq2seq_rows, :, :]
            seq2seq_chemo_iv = seq2seq_chemo_iv[:total_seq2seq_rows, :, :]
            seq2seq_radio_iv = seq2seq_radio_iv[:total_seq2seq_rows, :, :]
            seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]
            if seq2seq_stabilized_weights is not None:
                seq2seq_stabilized_weights = seq2seq_stabilized_weights[:total_seq2seq_rows]
            logger.info(f'total_seq2seq_rows: {total_seq2seq_rows}@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

            # Package outputs
            seq2seq_data = {
                'init_state': seq2seq_state_inits,
                'original_index': seq2seq_original_index,
                'active_encoder_r': seq2seq_active_encoder_r,
                'prev_treatments': seq2seq_previous_treatments,
                'current_treatments': seq2seq_current_treatments,
                'current_covariates': seq2seq_current_covariates,
                'prev_outputs': seq2seq_current_covariates[:, :, :1],
                'static_features': seq2seq_current_covariates[:, 0, 1:],
                'outputs': seq2seq_outputs,
                'U': seq2seq_U,
                'chemo_iv': seq2seq_chemo_iv,
                'radio_iv': seq2seq_radio_iv,
                'sequence_lengths': seq2seq_sequence_lengths,
                'active_entries': seq2seq_active_entries,
                'unscaled_outputs': seq2seq_outputs * self.scaling_params['output_stds'] + self.scaling_params['output_means'],
            }
            if seq2seq_stabilized_weights is not None:
                seq2seq_data['stabilized_weights'] = seq2seq_stabilized_weights

            self.data_original = deepcopy(self.data)
            self.data = seq2seq_data
            data_shapes = {k: v.shape for k, v in self.data.items()}
            #logger.info(f'func {inspect.currentframe().f_code.co_name} Shape of processed {self.subset_name} data: {data_shapes}')

            if save_encoder_r:
                self.encoder_r = encoder_r[:, :seq_length, :]

            self.processed_sequential = True
            self.exploded = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (multiple sequences)')

        return self.data

    def process_sequential_test(self, projection_horizon, encoder_r=None, save_encoder_r=False):
        logger.info(f'seqqqqqqqqqqqqqqqqqqqqqqqqqqq testttttttttttttttttttttttttttt')
        """
        Pre-process test dataset for multiple-step-ahead prediction: takes the last n-steps according to the projection horizon
        Args:
            projection_horizon: Projection horizon
            encoder_r: Representations of encoder
            save_encoder_r: Save all encoder representations (for cross-attention of EDCT)
        """

        assert self.processed

        if not self.processed_sequential:
            logger.info(f'Processing {self.subset_name} dataset before testing (multiple sequences)')

            sequence_lengths = self.data['sequence_lengths']
            outputs = self.data['outputs']
            current_treatments = self.data['current_treatments']
            previous_treatments = self.data['prev_treatments'][:, 1:, :]  # Without zero_init_treatment
            current_covariates = self.data['current_covariates']

            U = self.data['U']
            chemo_iv = self.data['chemo_iv']
            radio_iv = self.data['radio_iv']

            num_patient_points, max_seq_length, num_features = outputs.shape
            logger.info(f'num_patient_points: {num_patient_points}~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

            if encoder_r is not None:
                seq2seq_state_inits = np.zeros((num_patient_points, encoder_r.shape[-1]))
            seq2seq_active_encoder_r = np.zeros((num_patient_points, max_seq_length - projection_horizon))
            seq2seq_previous_treatments = np.zeros((num_patient_points, projection_horizon, previous_treatments.shape[-1]))
            seq2seq_current_treatments = np.zeros((num_patient_points, projection_horizon, current_treatments.shape[-1]))
            seq2seq_current_covariates = np.zeros((num_patient_points, projection_horizon, current_covariates.shape[-1]))
            seq2seq_outputs = np.zeros((num_patient_points, projection_horizon, outputs.shape[-1]))
            seq2seq_active_entries = np.zeros((num_patient_points, projection_horizon, 1))
            if self.latent_confounder:
                seq2seq_U = np.zeros((num_patient_points, projection_horizon, U.shape[-1]))
                seq2seq_chemo_iv = np.zeros((num_patient_points, projection_horizon, chemo_iv.shape[-1]))
                seq2seq_radio_iv = np.zeros((num_patient_points, projection_horizon, radio_iv.shape[-1]))
            seq2seq_sequence_lengths = np.zeros(num_patient_points)

            for i in range(num_patient_points):
            #!!!!!!!!!!!!!!!!!!!!!!!
            #for i in range(min(num_patient_points, U.shape[0]-1)):
                fact_length = int(sequence_lengths[i]) - projection_horizon
                if encoder_r is not None:
                    seq2seq_state_inits[i] = encoder_r[i, fact_length - 1]
                seq2seq_active_encoder_r[i, :fact_length] = 1.0

                seq2seq_active_entries[i] = np.ones(shape=(projection_horizon, 1))
                seq2seq_previous_treatments[i] = previous_treatments[i, fact_length - 1:fact_length + projection_horizon - 1, :]
                seq2seq_current_treatments[i] = current_treatments[i, fact_length:fact_length + projection_horizon, :]
                seq2seq_outputs[i] = outputs[i, fact_length: fact_length + projection_horizon, :]
                seq2seq_sequence_lengths[i] = projection_horizon
                # Disabled teacher forcing for test dataset
                seq2seq_current_covariates[i] = np.repeat([current_covariates[i, fact_length - 1]], projection_horizon, axis=0)
                if self.latent_confounder:
                    seq2seq_U[i] = np.repeat([U[i, fact_length - 1]], projection_horizon, axis=0)
                    seq2seq_chemo_iv[i] = np.repeat([chemo_iv[i, fact_length - 1]], projection_horizon, axis=0)
                    seq2seq_radio_iv[i] = np.repeat([radio_iv[i, fact_length - 1]], projection_horizon, axis=0)
                else:
                    seq2seq_U = None
                    seq2seq_chemo_iv = None
                    seq2seq_radio_iv = None

            # Package outputs
            seq2seq_data = {
                'active_encoder_r': seq2seq_active_encoder_r,
                'prev_treatments': seq2seq_previous_treatments,
                'current_treatments': seq2seq_current_treatments,
                'current_covariates': seq2seq_current_covariates,
                'prev_outputs': seq2seq_current_covariates[:, :, :1],
                'static_features': seq2seq_current_covariates[:, 0, 1:],
                'outputs': seq2seq_outputs,
                'U': seq2seq_U,
                'chemo_iv': seq2seq_chemo_iv,
                'radio_iv': seq2seq_radio_iv,
                'sequence_lengths': seq2seq_sequence_lengths,
                'active_entries': seq2seq_active_entries,
                'unscaled_outputs': seq2seq_outputs * self.scaling_params['output_stds'] + self.scaling_params['output_means'],
                'patient_types': self.data['patient_types'],
                'patient_ids_all_trajectories': self.data['patient_ids_all_trajectories'],
                'patient_current_t': self.data['patient_current_t']
            }
            if encoder_r is not None:
                seq2seq_data['init_state'] = seq2seq_state_inits

            self.data_original = deepcopy(self.data)
            self.data = seq2seq_data

            process_seq_test_data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'func {inspect.currentframe().f_code.co_name} Shape of processed {self.subset_name} process_seq_test_data_shapes: {process_seq_test_data_shapes}')

            if save_encoder_r and encoder_r is not None:
                self.encoder_r = encoder_r[:, :max_seq_length - projection_horizon, :]

            self.processed_sequential = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (multiple sequences)')

        return self.data

    def process_autoregressive_test(self, encoder_r, encoder_outputs, projection_horizon, save_encoder_r=False):
        """
        Pre-process test dataset for multiple-step-ahead prediction: axillary dataset placeholder for autoregressive prediction
        Args:
            projection_horizon: Projection horizon
            encoder_r: Representations of encoder
            save_encoder_r: Save all encoder representations (for cross-attention of EDCT)
        """

        assert self.processed_sequential

        if not self.processed_autoregressive:
            logger.info(f'Processing {self.subset_name} dataset before testing (autoregressive)')

            current_treatments = self.data_original['current_treatments']
            prev_treatments = self.data_original['prev_treatments'][:, 1:, :]  # Without zero_init_treatment

            U = self.data_original['U']
            chemo_iv = self.data_original['chemo_iv']
            radio_iv = self.data_original['radio_iv']
            sequence_lengths = self.data_original['sequence_lengths']
            num_patient_points, max_seq_length = current_treatments.shape[:2]
            logger.info(f'num_patient_points: {num_patient_points}######################################')

            current_dataset = dict()  # Same as original, but only with last n-steps
            current_dataset['current_covariates'] = np.zeros((num_patient_points, projection_horizon,
                                                              self.data_original['current_covariates'].shape[-1]))
            current_dataset['prev_treatments'] = np.zeros((num_patient_points, projection_horizon,
                                                           self.data_original['prev_treatments'].shape[-1]))
            current_dataset['current_treatments'] = np.zeros((num_patient_points, projection_horizon,
                                                              self.data_original['current_treatments'].shape[-1]))
            current_dataset['init_state'] = np.zeros((num_patient_points, encoder_r.shape[-1]))
            current_dataset['active_encoder_r'] = np.zeros((num_patient_points, max_seq_length - projection_horizon))
            current_dataset['active_entries'] = np.ones((num_patient_points, projection_horizon, 1))
            if self.latent_confounder:
                current_dataset['U'] = np.zeros((num_patient_points, projection_horizon, U.shape[-1]))
                current_dataset['chemo_iv'] = np.zeros((num_patient_points, projection_horizon, chemo_iv.shape[-1]))
                current_dataset['radio_iv'] = np.zeros((num_patient_points, projection_horizon, radio_iv.shape[-1]))

            for i in range(num_patient_points):
                fact_length = int(sequence_lengths[i]) - projection_horizon
                current_dataset['init_state'][i] = encoder_r[i, fact_length - 1]
                current_dataset['current_covariates'][i, 0, 0] = encoder_outputs[i, fact_length - 1]
                current_dataset['active_encoder_r'][i, :fact_length] = 1.0
                current_dataset['prev_treatments'][i] = \
                    prev_treatments[i, fact_length - 1:fact_length + projection_horizon - 1, :]
                current_dataset['current_treatments'][i] = current_treatments[i, fact_length:fact_length + projection_horizon, :]
                if self.latent_confounder:
                    current_dataset['U'][i] = np.repeat([U[i, fact_length - 1]], projection_horizon, axis=0)
                    current_dataset['chemo_iv'][i] = np.repeat([chemo_iv[i, fact_length - 1]], projection_horizon, axis=0)
                    current_dataset['radio_iv'][i] = np.repeat([radio_iv[i, fact_length - 1]], projection_horizon, axis=0)
                else:
                    current_dataset['U'] = None
                    current_dataset['chemo_iv'] = None
                    current_dataset['radio_iv'] = None

            current_dataset['prev_outputs'] = current_dataset['current_covariates'][:, :, :1]
            current_dataset['static_features'] = self.data_original['static_features']

            self.data_processed_seq = deepcopy(self.data)
            self.data = current_dataset
            #data_shapes = {k: v.shape for k, v in self.data.items()}
            #logger.info(f'func {inspect.currentframe().f_code.co_name} Shape of processed {self.subset_name} data: {data_shapes}')

            if save_encoder_r:
                self.encoder_r = encoder_r[:, :max_seq_length - projection_horizon, :]

            self.processed_autoregressive = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (autoregressive)')

        return self.data

    def process_sequential_multi(self, projection_horizon):
        logger.info(f'mullllllllllllllllllllllllllllllllllll')
        """
        Pre-process test dataset for multiple-step-ahead prediction for multi-input model: marking rolling origin with
            'future_past_split'
        Args:
            projection_horizon: Projection horizon
        """

        assert self.processed_sequential

        if not self.processed_autoregressive:
            self.data_processed_seq = self.data
            self.data = deepcopy(self.data_original)
            self.data['future_past_split'] = self.data['sequence_lengths'] - projection_horizon

            self.processed_autoregressive = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (autoregressive)')

        return self.data


class SyntheticCancerDatasetCollection(SyntheticDatasetCollection):
    """
    Dataset collection (train_f, val_f, test_cf_one_step, test_cf_treatment_seq)
    """

    def __init__(self,
                 chemo_coeff: float,
                 radio_coeff: float,
                 num_patients: dict,
                 latent_confounder: bool,
                 latent_confounder_influence_ratio: float,
                 chemo_iv_influence_ratio: float,
                 radio_iv_influence_ratio: float,
                 seed=100,
                 window_size=15,
                 max_seq_length=60,
                 projection_horizon=5,
                 lag=0,
                 cf_seq_mode='sliding_treatment',
                 treatment_mode='multiclass',
                 **kwargs):
        """
        Args:
            chemo_coeff: Confounding coefficient of chemotherapy
            radio_coeff: Confounding coefficient of radiotherapy
            num_patients: Number of patients in dataset
            window_size: Used for biased treatment assignment
            max_seq_length: Max length of time series
            subset_name: train / val / test
            mode: factual / counterfactual_one_step / counterfactual_treatment_seq
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            seed: Random seed
            lag: Lag for treatment assignment window
            cf_seq_mode: sliding_treatment / random_trajectories
            treatment_mode: multiclass / multilabel
        """
        super(SyntheticCancerDatasetCollection, self).__init__()
        self.latent_confounder = latent_confounder
        self.latent_confounder_influence_ratio = latent_confounder_influence_ratio
        self.chemo_iv_influence_ratio = chemo_iv_influence_ratio
        self.radio_iv_influence_ratio = radio_iv_influence_ratio
        self.seed = seed
        np.random.seed(seed)

        self.train_f = SyntheticCancerDataset(chemo_coeff, radio_coeff, num_patients['train'], window_size, max_seq_length,
                                              'train', lag=lag, treatment_mode=treatment_mode, latent_confounder=latent_confounder,
                                              u_ratio=latent_confounder_influence_ratio, chemo_iv_ratio=chemo_iv_influence_ratio, radio_iv_ratio=radio_iv_influence_ratio)
        self.val_f = SyntheticCancerDataset(chemo_coeff, radio_coeff, num_patients['val'], window_size, max_seq_length, 'val',
                                            lag=lag, treatment_mode=treatment_mode, latent_confounder=latent_confounder,
                                            u_ratio = latent_confounder_influence_ratio, chemo_iv_ratio = chemo_iv_influence_ratio, radio_iv_ratio = radio_iv_influence_ratio)
        self.test_cf_one_step = SyntheticCancerDataset(chemo_coeff, radio_coeff, num_patients['test'], window_size,
                                                       max_seq_length, 'test', mode='counterfactual_one_step', lag=lag,
                                                       treatment_mode=treatment_mode, latent_confounder=latent_confounder,
                                                       u_ratio = latent_confounder_influence_ratio, chemo_iv_ratio = chemo_iv_influence_ratio, radio_iv_ratio = radio_iv_influence_ratio)
        self.test_cf_treatment_seq = SyntheticCancerDataset(chemo_coeff, radio_coeff, num_patients['test'], window_size,
                                                            max_seq_length, 'test', mode='counterfactual_treatment_seq',
                                                            projection_horizon=projection_horizon, lag=lag,
                                                            cf_seq_mode=cf_seq_mode, treatment_mode=treatment_mode, latent_confounder=latent_confounder,
                                                            u_ratio = latent_confounder_influence_ratio, chemo_iv_ratio = chemo_iv_influence_ratio, radio_iv_ratio = radio_iv_influence_ratio)
        '''class_name = self.__class__.__name__
        file_name = __file__
        logger.info(f'Class: {class_name}, File: {file_name}')
        logger.info(f'@@train_f keys: {list(self.train_f.data.keys())}')
        logger.info(f'@@val_f keys: {list(self.val_f.data.keys())}')
        logger.info(f'@@test_cf_one_step keys: {list(self.test_cf_one_step.data.keys())}')
        logger.info(f'@@test_cf_treatment_seq keys: {list(self.test_cf_treatment_seq.data.keys())}')'''
        self.projection_horizon = projection_horizon
        self.autoregressive = True
        self.has_vitals = False
        self.train_scaling_params = self.train_f.get_scaling_params()
