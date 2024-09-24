from pytorch_lightning import LightningModule
from omegaconf import DictConfig
import torch
from torch import nn
from omegaconf.errors import MissingMandatoryValue
import torch.nn.functional as F
from hydra.utils import instantiate
from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader, Dataset, Subset
from src.models.utils_transformer import TransformerEncoderBlock, TransformerDecoderBlock, AbsolutePositionalEncoding, \
    RelativePositionalEncoding
from src.models.utils import grad_reverse
import logging
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from functools import partial
import seaborn as sns
from sklearn.manifold import TSNE

from src.models.edct import EDCT
from src.models.time_varying_model import BRCausalModel
from src.models.utils_transformer import TransformerMultiInputBlock, IVTransformerMultiInputBlock, OnlyIVTransformerMultiInputBlock, LayerNorm
from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.utils import BRTreatmentOutcomeHead, bce

logger = logging.getLogger(__name__)

class FirstStageHead(nn.Module):
    def __init__(self, seq_hidden_units, br_size, fc_hidden_units, dim_treatments, dim_outcome, dim_iv, balancing='grad_reverse'):
        super().__init__()

        self.seq_hidden_units = seq_hidden_units
        self.br_size = br_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome
        self.dim_iv = dim_iv
        self.balancing = balancing

        self.linear1 = nn.Linear(self.seq_hidden_units, self.br_size)
        self.elu1 = nn.ELU()

        self.linear2 = nn.Linear(self.seq_hidden_units, self.dim_iv)
        self.elu2 = nn.ELU()

        self.linear3 = nn.Linear(self.br_size + self.dim_iv, self.fc_hidden_units)
        self.elu3 = nn.ELU()
        self.linear4 = nn.Linear(self.fc_hidden_units, self.dim_treatments)

        self.treatment_head_params = ['linear1', 'linear2', 'linear3', 'linear4']

    def my_build_treatment(self, br, iv, detached=False):
        if detached:
            br = br.detach()

        logger.info(f'my build treatment 00000000000000000000')
        if self.balancing == 'grad_reverse':
            br = grad_reverse(br, self.alpha)

        x = torch.cat((br, iv), dim=-1)
        x = self.elu3(self.linear3(x))
        treatment = self.linear4(x)  # Softmax is encapsulated into F.cross_entropy()
        return treatment

    def first_stage_build_br(self, seq_output):
        br = self.elu1(self.linear1(seq_output))
        return br

    def my_build_iv(self, seq_output):
        iv = self.elu2(self.linear2(seq_output))
        return iv

class SecondStageHead(nn.Module):
    def __init__(self, seq_hidden_units, br_size, fc_hidden_units, dim_treatments, dim_outcome):
        super().__init__()

        self.seq_hidden_units = seq_hidden_units
        self.br_size = br_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome

        self.linear1 = nn.Linear(self.br_size + self.dim_treatments, self.fc_hidden_units)
        self.elu1 = nn.ELU()
        self.linear2 = nn.Linear(self.fc_hidden_units, self.dim_outcome)

        self.linear3 = nn.Linear(self.seq_hidden_units, self.br_size)
        self.elu3 = nn.ELU()

        #self.treatment_head_params = ['linear2', 'linear3']

    def second_stage_build_br(self, seq_output):
        br = self.elu3(self.linear3(seq_output))
        return br

    def build_outcome(self, br, current_treatment):
        x = torch.cat((br, current_treatment), dim=-1)
        x = self.elu1(self.linear1(x))
        outcome = self.linear2(x)
        return outcome


#class IVCT(EDCT):
class IVCT(BRCausalModel):
    model_type = 'iv_multi'
    possible_model_types = ['iv_multi']
    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 projection_horizon: int = None,
                 bce_weights: np.array = None,
                 alpha: float = 0.0,
                 update_alpha: bool=True,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        print("IVCT __init__ args.model:", args.model)
        if self.dataset_collection is not None:
            self.projection_horizon = self.dataset_collection.projection_horizon
        else:
            self.projection_horizon = projection_horizon

        # Used in hparam tuning
        self.input_size = max(self.dim_treatments, self.dim_static_features, self.dim_vitals, self.dim_outcome, self.dim_iv)
        logger.info(f'Max input size of {self.model_type}: {self.input_size}')
        assert self.autoregressive  # prev_outcomes are obligatory

        self.basic_block_cls = TransformerMultiInputBlock
        self._init_specific(args.model.iv_multi)
        self.first_stage_output_dropout = nn.Dropout(self.dropout_rate)
        self.iv_dropout = nn.Dropout(self.dropout_rate)
        self.second_stage_output_dropout = nn.Dropout(self.dropout_rate)

        self.balancing = args.exp.balancing
        self.alpha = alpha if not update_alpha else 0.0
        self.alpha_max = alpha
        self.save_hyperparameters(args)
        '''if self.hparams.exp.weights_ema:
            self.ema_treatment = ExponentialMovingAverage([param for name, param in self.named_parameters()], decay=0.9)
            self.ema_non_treatment = ExponentialMovingAverage([param for name, param in self.named_parameters()], decay=0.9)'''
        #self.automatic_optimization = False

    def init_first_stage(self, sub_args: DictConfig):
        self.first_stage_treatments_input_transformation = nn.Linear(self.dim_treatments, self.seq_hidden_units)
        self.first_stage_vitals_input_transformation = \
            nn.Linear(self.dim_vitals, self.seq_hidden_units) if self.has_vitals else None
        self.first_stage_vitals_input_transformation = nn.Linear(self.dim_vitals, self.seq_hidden_units) if self.has_vitals else None
        self.first_stage_outputs_input_transformation = nn.Linear(self.dim_outcome, self.seq_hidden_units)
        self.first_stage_static_input_transformation = nn.Linear(self.dim_static_features, self.seq_hidden_units)
        self.chemo_iv_input_transformation = nn.Linear(self.dim_iv, self.seq_hidden_units)
        self.radio_iv_input_transformation = nn.Linear(self.dim_iv, self.seq_hidden_units)

        self.first_stage_self_positional_encoding = self.first_stage_self_positional_encoding_k = self.first_stage_self_positional_encoding_v = None
        if sub_args.self_positional_encoding.absolute:
            self.first_stage_self_positional_encoding = \
                AbsolutePositionalEncoding(self.max_seq_length, self.seq_hidden_units,
                                           sub_args.self_positional_encoding.trainable)
        else:
            # Relative positional encoding is shared across heads
            self.first_stage_self_positional_encoding_k = \
                RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.self_positional_encoding.trainable)
            self.first_stage_self_positional_encoding_v = \
                RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.self_positional_encoding.trainable)

        self.first_stage_cross_positional_encoding = self.first_stage_cross_positional_encoding_k = self.first_stage_cross_positional_encoding_v = None
        if 'cross_positional_encoding' in sub_args and sub_args.cross_positional_encoding.absolute:
            self.first_stage_cross_positional_encoding = \
                AbsolutePositionalEncoding(self.max_seq_length, self.seq_hidden_units,
                                           sub_args.cross_positional_encoding.trainable)
        elif 'cross_positional_encoding' in sub_args:
            # Relative positional encoding is shared across heads
            self.first_stage_cross_positional_encoding_k = \
                RelativePositionalEncoding(sub_args.cross_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.cross_positional_encoding.trainable, cross_attn=True)
            self.first_stage_cross_positional_encoding_v = \
                RelativePositionalEncoding(sub_args.cross_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.cross_positional_encoding.trainable, cross_attn=True)

        self.first_stage_transformer_blocks = nn.ModuleList(
            [self.basic_block_cls(self.seq_hidden_units, self.num_heads, self.head_size, self.seq_hidden_units * 4,
                                  self.dropout_rate,
                                  self.dropout_rate if sub_args.attn_dropout else 0.0,
                                  self_positional_encoding_k=self.first_stage_self_positional_encoding_k,
                                  self_positional_encoding_v=self.first_stage_self_positional_encoding_v,
                                  n_inputs=self.n_inputs,
                                  disable_cross_attention=sub_args.disable_cross_attention,
                                  isolate_subnetwork=sub_args.isolate_subnetwork) for _ in range(self.num_layer)])

        self.iv_self_positional_encoding = self.iv_self_positional_encoding_k = self.iv_self_positional_encoding_v = None
        if sub_args.self_positional_encoding.absolute:
            self.iv_self_positional_encoding = \
                AbsolutePositionalEncoding(self.max_seq_length, self.seq_hidden_units,
                                           sub_args.self_positional_encoding.trainable)
        else:
            # Relative positional encoding is shared across heads
            self.iv_self_positional_encoding_k = \
                RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.self_positional_encoding.trainable)
            self.iv_self_positional_encoding_v = \
                RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.self_positional_encoding.trainable)

        self.iv_block_cls = OnlyIVTransformerMultiInputBlock
        self.onlyiv_transformer_blocks = nn.ModuleList(
            [self.iv_block_cls(self.seq_hidden_units, self.num_heads, self.head_size, self.seq_hidden_units * 4,
                               self.dropout_rate,
                               self.dropout_rate if sub_args.attn_dropout else 0.0,
                               self_positional_encoding_k=self.iv_self_positional_encoding_k,
                               self_positional_encoding_v=self.iv_self_positional_encoding_v,
                               n_inputs=self.n_iv_inputs,
                               disable_cross_attention=sub_args.disable_cross_attention,
                               isolate_subnetwork=sub_args.isolate_subnetwork) for _ in range(self.num_layer)])

        self.first_stage_head = FirstStageHead(self.seq_hidden_units, self.br_size,
                                               self.fc_hidden_units, self.dim_treatments,
                                               self.dim_outcome, self.dim_iv, self.balancing)


    def init_second_stage(self, sub_args: DictConfig):
        self.second_stage_treatments_input_transformation = nn.Linear(self.dim_treatments, self.seq_hidden_units)
        self.second_stage_vitals_input_transformation = \
            nn.Linear(self.dim_vitals, self.seq_hidden_units) if self.has_vitals else None
        self.second_stage_vitals_input_transformation = nn.Linear(self.dim_vitals, self.seq_hidden_units) if self.has_vitals else None
        self.second_stage_outputs_input_transformation = nn.Linear(self.dim_outcome, self.seq_hidden_units)
        self.second_stage_static_input_transformation = nn.Linear(self.dim_static_features, self.seq_hidden_units)

        self.second_stage_self_positional_encoding = self.second_stage_self_positional_encoding_k = self.second_stage_self_positional_encoding_v = None
        if sub_args.self_positional_encoding.absolute:
            self.second_stage_self_positional_encoding = \
                AbsolutePositionalEncoding(self.max_seq_length, self.seq_hidden_units,
                                           sub_args.self_positional_encoding.trainable)
        else:
            # Relative positional encoding is shared across heads
            self.second_stage_self_positional_encoding_k = \
                RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.self_positional_encoding.trainable)
            self.second_stage_self_positional_encoding_v = \
                RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.self_positional_encoding.trainable)

        self.second_stage_cross_positional_encoding = self.second_stage_cross_positional_encoding_k = self.second_stage_cross_positional_encoding_v = None
        if 'cross_positional_encoding' in sub_args and sub_args.cross_positional_encoding.absolute:
            self.second_stage_cross_positional_encoding = \
                AbsolutePositionalEncoding(self.max_seq_length, self.seq_hidden_units,
                                           sub_args.cross_positional_encoding.trainable)
        elif 'cross_positional_encoding' in sub_args:
            # Relative positional encoding is shared across heads
            self.second_stage_cross_positional_encoding_k = \
                RelativePositionalEncoding(sub_args.cross_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.cross_positional_encoding.trainable, cross_attn=True)
            self.second_stage_cross_positional_encoding_v = \
                RelativePositionalEncoding(sub_args.cross_positional_encoding.max_relative_position, self.head_size,
                                           sub_args.cross_positional_encoding.trainable, cross_attn=True)

        self.second_stage_transformer_blocks = nn.ModuleList(
            [self.basic_block_cls(self.seq_hidden_units, self.num_heads, self.head_size, self.seq_hidden_units * 4,
                                  self.dropout_rate,
                                  self.dropout_rate if sub_args.attn_dropout else 0.0,
                                  self_positional_encoding_k=self.second_stage_self_positional_encoding_k,
                                  self_positional_encoding_v=self.second_stage_self_positional_encoding_v,
                                  n_inputs=self.n_inputs,
                                  disable_cross_attention=sub_args.disable_cross_attention,
                                  isolate_subnetwork=sub_args.isolate_subnetwork) for _ in range(self.num_layer)])

        self.second_stage_head = SecondStageHead(self.seq_hidden_units, self.br_size, self.fc_hidden_units,
                                                 self.dim_treatments, self.dim_outcome)


    def _init_specific(self, sub_args: DictConfig):
        """
        Initialization of specific sub-network (only multi)
        Args:
            sub_args: sub-network hyperparameters
        """
        try:
            #super(IVCT, self)._init_specific(sub_args)
            #------------------------------------
            self.max_seq_length = sub_args.max_seq_length
            self.br_size = sub_args.br_size  # balanced representation size
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.dropout_rate = sub_args.dropout_rate
            # self.attn_dropout_rate = sub_args.attn_dropout_rate

            self.num_layer = sub_args.num_layer
            self.num_heads = sub_args.num_heads
            #------------------------------------

            if self.seq_hidden_units is None or self.br_size is None or self.fc_hidden_units is None \
                    or self.dropout_rate is None:
                raise MissingMandatoryValue()

            self.head_size = sub_args.seq_hidden_units // sub_args.num_heads

            self.n_inputs = 3 if self.has_vitals else 2  # prev_outcomes and prev_treatments
            self.n_iv_inputs = 2
            self.init_first_stage(sub_args)
            self.init_second_stage(sub_args)

            # self.last_layer_norm = LayerNorm(self.seq_hidden_units)
        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()


    def first_stage_build_br(self, prev_treatments, vitals, prev_outputs, static_features, active_entries, fixed_split):

        active_entries_treat_outcomes = torch.clone(active_entries)
        active_entries_vitals = torch.clone(active_entries)

        if fixed_split is not None and self.has_vitals:  # Test sequence data / Train augmented data
            for i in range(len(active_entries)):

                # Masking vitals in range [fixed_split: ]
                active_entries_vitals[i, int(fixed_split[i]):, :] = 0.0
                vitals[i, int(fixed_split[i]):] = 0.0

        x_t = self.first_stage_treatments_input_transformation(prev_treatments)
        x_o = self.first_stage_outputs_input_transformation(prev_outputs)
        x_v = self.first_stage_vitals_input_transformation(vitals) if self.has_vitals else None
        x_s = self.first_stage_static_input_transformation(static_features.unsqueeze(1))  # .expand(-1, x_t.size(1), -1)

        # if active_encoder_br is None and encoder_r is None:  # Only self-attention
        for block in self.first_stage_transformer_blocks:

            if self.first_stage_self_positional_encoding is not None:
                x_t = x_t + self.first_stage_self_positional_encoding(x_t)
                x_o = x_o + self.first_stage_self_positional_encoding(x_o)
                x_v = x_v + self.first_stage_self_positional_encoding(x_v) if self.has_vitals else None

            if self.has_vitals:
                x_t, x_o, x_v = block((x_t, x_o, x_v), x_s, active_entries_treat_outcomes, active_entries_vitals)
            else:
                x_t, x_o = block((x_t, x_o), x_s, active_entries_treat_outcomes)

        if not self.has_vitals:
            x = (x_o + x_t) / 2
        else:
            if fixed_split is not None:  # Test seq data
                x = torch.empty_like(x_o)
                for i in range(len(active_entries)):
                    # Masking vitals in range [fixed_split: ]
                    x[i, :int(fixed_split[i])] = \
                        (x_o[i, :int(fixed_split[i])] + x_t[i, :int(fixed_split[i])] + x_v[i, :int(fixed_split[i])]) / 3
                    x[i, int(fixed_split[i]):] = (x_o[i, int(fixed_split[i]):] + x_t[i, int(fixed_split[i]):]) / 2
            else:  # Train data always has vitals
                x = (x_o + x_t + x_v) / 3

        output = self.first_stage_output_dropout(x)
        br = self.first_stage_head.first_stage_build_br(output)
        return br

    def first_stage_build_iv(self, active_entries, chemo_iv, radio_iv, fixed_split):

        active_entries_treat_outcomes = torch.clone(active_entries)
        active_entries_vitals = torch.clone(active_entries)

        if fixed_split is not None and self.has_vitals:  # Test sequence data / Train augmented data
            for i in range(len(active_entries)):

                # Masking vitals in range [fixed_split: ]
                active_entries_vitals[i, int(fixed_split[i]):, :] = 0.0

        x_chemo_iv = self.chemo_iv_input_transformation(chemo_iv)
        x_radio_iv = self.radio_iv_input_transformation(radio_iv)

        # if active_encoder_br is None and encoder_r is None:  # Only self-attention
        for block in self.onlyiv_transformer_blocks:

            if self.iv_self_positional_encoding is not None:
                x_chemo_iv = x_chemo_iv + self.iv_self_positional_encoding(x_chemo_iv)
                x_radio_iv = x_radio_iv + self.iv_self_positional_encoding(x_radio_iv)

            '''
            if self.has_vitals:
                x_chemo_iv, x_radio_iv, x_v = block((x_chemo_iv, x_radio_iv, x_v), active_entries_treat_outcomes, active_entries_vitals)
            else:
                x_chemo_iv, x_radio_iv = block((x_chemo_iv, x_radio_iv), active_entries_treat_outcomes)
            '''
            x_chemo_iv, x_radio_iv = block((x_chemo_iv, x_radio_iv), active_entries_treat_outcomes)

        if not self.has_vitals:
            x = (x_chemo_iv + x_radio_iv) / 2
        else:
            if fixed_split is not None:  # Test seq data
                x = torch.empty_like(x_chemo_iv)
                for i in range(len(active_entries)):
                    # Masking vitals in range [fixed_split: ]
                    x[i, :int(fixed_split[i])] = \
                        (x_chemo_iv[i, :int(fixed_split[i])] + x_radio_iv[i, :int(fixed_split[i])]) / 3
                    x[i, int(fixed_split[i]):] = (x_chemo_iv[i, int(fixed_split[i]):] + x_radio_iv[i, int(fixed_split[i]):]) / 2
            else:
                x = (x_chemo_iv + x_radio_iv) / 2

        iv = self.iv_dropout(x)
        iv = self.first_stage_head.my_build_iv(iv) #把br_treatment_outcome_head 用作iv的,其他两个分别第一和第二阶段的各用各的(first_stage_br_treatment_outcome_head,second_stage_br_treatment_outcome_head )
        return iv

    def second_stage_build_br(self, prev_treatments, vitals, prev_outputs, static_features, active_entries, fixed_split):

        active_entries_treat_outcomes = torch.clone(active_entries)
        active_entries_vitals = torch.clone(active_entries)

        if fixed_split is not None and self.has_vitals:  # Test sequence data / Train augmented data
            for i in range(len(active_entries)):

                # Masking vitals in range [fixed_split: ]
                active_entries_vitals[i, int(fixed_split[i]):, :] = 0.0
                vitals[i, int(fixed_split[i]):] = 0.0

        x_t = self.second_stage_treatments_input_transformation(prev_treatments)
        x_o = self.second_stage_outputs_input_transformation(prev_outputs)
        x_v = self.second_stage_vitals_input_transformation(vitals) if self.has_vitals else None
        x_s = self.second_stage_static_input_transformation(static_features.unsqueeze(1))  # .expand(-1, x_t.size(1), -1)

        # if active_encoder_br is None and encoder_r is None:  # Only self-attention
        for block in self.second_stage_transformer_blocks:

            if self.second_stage_self_positional_encoding is not None:
                x_t = x_t + self.second_stage_self_positional_encoding(x_t)
                x_o = x_o + self.second_stage_self_positional_encoding(x_o)
                x_v = x_v + self.second_stage_self_positional_encoding(x_v) if self.has_vitals else None

            if self.has_vitals:
                x_t, x_o, x_v = block((x_t, x_o, x_v), x_s, active_entries_treat_outcomes, active_entries_vitals)
            else:
                x_t, x_o = block((x_t, x_o), x_s, active_entries_treat_outcomes)

        if not self.has_vitals:
            x = (x_o + x_t) / 2
        else:
            if fixed_split is not None:  # Test seq data
                x = torch.empty_like(x_o)
                for i in range(len(active_entries)):
                    # Masking vitals in range [fixed_split: ]
                    x[i, :int(fixed_split[i])] = \
                        (x_o[i, :int(fixed_split[i])] + x_t[i, :int(fixed_split[i])] + x_v[i, :int(fixed_split[i])]) / 3
                    x[i, int(fixed_split[i]):] = (x_o[i, int(fixed_split[i]):] + x_t[i, int(fixed_split[i]):]) / 2
            else:  # Train data always has vitals
                x = (x_o + x_t + x_v) / 3

        output = self.second_stage_output_dropout(x)
        br = self.second_stage_head.second_stage_build_br(output)
        return br
    def get_autoregressive_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Autoregressive Prediction for {dataset.subset_name}.')

        predicted_outputs = np.zeros((len(dataset), self.hparams.dataset.projection_horizon, self.dim_outcome))

        for t in range(self.hparams.dataset.projection_horizon + 1):
            logger.info(f't = {t + 1}')
            outputs_scaled = self.get_predictions(dataset)

            for i in range(len(dataset)):
                split = int(dataset.data['future_past_split'][i])
                if t < self.hparams.dataset.projection_horizon:
                    dataset.data['prev_outputs'][i, split + t, :] = outputs_scaled[i, split - 1 + t, :]
                if t > 0:
                    predicted_outputs[i, t - 1, :] = outputs_scaled[i, split - 1 + t, :]

        return predicted_outputs

    def visualize(self, dataset: Dataset, index=0, artifacts_path=None):
        """
        Vizualizes attention scores
        :param dataset: dataset
        :param index: index of an instance
        :param artifacts_path: Path for saving
        """
        fig_keys = ['self_attention_o', 'self_attention_t', 'cross_attention_ot', 'cross_attention_to']
        if self.has_vitals:
            fig_keys += ['cross_attention_vo', 'cross_attention_ov', 'cross_attention_vt', 'cross_attention_tv',
                         'self_attention_v']
        self._visualize(fig_keys, dataset, index, artifacts_path)


    def get_predictions_first_stage(self, batch):
        logger.debug(f'get_predictions_first_stage---------')
        fixed_split = batch['future_past_split'] if 'future_past_split' in batch else None

        if self.training and self.hparams.model.iv_multi.augment_with_masked_vitals and self.has_vitals:
            # Augmenting original batch with vitals-masked copy
            assert fixed_split is None  # Only for training data
            fixed_split = torch.empty((2 * len(batch['active_entries']),)).type_as(batch['active_entries'])
            for i, seq_len in enumerate(batch['active_entries'].sum(1).int()):
                fixed_split[i] = seq_len  # Original batch
                fixed_split[len(batch['active_entries']) + i] = torch.randint(0, int(seq_len) + 1, (1,)).item()  # Augmented batch

            for (k, v) in batch.items():
                batch[k] = torch.cat((v, v), dim=0)

        prev_treatments = batch['prev_treatments']
        vitals = batch['vitals'] if self.has_vitals else None
        prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        active_entries = batch['active_entries']
        chemo_iv = batch['chemo_iv'] #if self.training else None
        radio_iv = batch['radio_iv'] #if self.training else None

        br = self.first_stage_build_br(prev_treatments, vitals, prev_outputs, static_features, active_entries, fixed_split)
        iv = self.first_stage_build_iv(active_entries, chemo_iv, radio_iv, fixed_split)
        treatment_pred = self.first_stage_head.my_build_treatment(br, iv)

        return treatment_pred

    def get_predictions_second_stage(self, batch, treatment_pred_or_real):
        logger.debug(f'get_predictions_second_stage---------')
        fixed_split = batch['future_past_split'] if 'future_past_split' in batch else None

        if self.training and self.hparams.model.iv_multi.augment_with_masked_vitals and self.has_vitals:
            # Augmenting original batch with vitals-masked copy
            assert fixed_split is None  # Only for training data
            fixed_split = torch.empty((2 * len(batch['active_entries']),)).type_as(batch['active_entries'])
            for i, seq_len in enumerate(batch['active_entries'].sum(1).int()):
                fixed_split[i] = seq_len  # Original batch
                fixed_split[len(batch['active_entries']) + i] = torch.randint(0, int(seq_len) + 1, (1,)).item()  # Augmented batch

            for (k, v) in batch.items():
                batch[k] = torch.cat((v, v), dim=0)

        prev_treatments = batch['prev_treatments']
        vitals = batch['vitals'] if self.has_vitals else None
        prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        active_entries = batch['active_entries']

        br = self.second_stage_build_br(prev_treatments, vitals, prev_outputs, static_features, active_entries, fixed_split)
        outcome_pred = self.second_stage_head.build_outcome(br, treatment_pred_or_real)

        #logger.info(f'treatment_pred shape: {treatment_pred.shape}, outcome_pred shape: {outcome_pred.shape}, br shape: {br.shape}.')
        return outcome_pred, br

    def configure_optimizers(self):
        #optimizer = self._get_optimizer(list(self.named_parameters()))
        #return optimizer
        if self.balancing == 'grad_reverse' and not self.hparams.exp.weights_ema:  # one optimizer
            optimizer = self._get_optimizer(list(self.named_parameters()))

            if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
                return self._get_lr_schedulers(optimizer)

            return optimizer

        else:
            treatment_head_names = [
                name for name, _ in self.named_parameters()
                if 'first_stage' in name or 'iv_self' in name or 'onlyiv_transformer_blocks' in name
            ]
            treatment_head_params = [
                (name, param) for name, param in self.named_parameters()
                if name in treatment_head_names
            ]
            non_treatment_head_params = [
                (name, param) for name, param in self.named_parameters()
                if name not in treatment_head_names
            ]
            assert len(treatment_head_params + non_treatment_head_params) == len(list(self.named_parameters()))

            all_head_params = [
                (name, param) for name, param in self.named_parameters()
            ]

            if self.hparams.exp.weights_ema:
                self.ema_treatment = ExponentialMovingAverage([par[1] for par in treatment_head_params],
                                                              decay=self.hparams.exp.beta)
                #self.ema_non_treatment = ExponentialMovingAverage([par[1] for par in non_treatment_head_params],
                #                                                  decay=self.hparams.exp.beta)
                self.ema_non_treatment = ExponentialMovingAverage([par[1] for par in all_head_params],#如果用了all_head_params,即所有参数，则optimizer是所有参数的optimizer
                                                                  decay=self.hparams.exp.beta)

            first_stage_optimizer = self._get_optimizer(treatment_head_params)
            second_stage_optimizer = self._get_optimizer(non_treatment_head_params) #如果用了all_head_params,即所有参数，则optimizer是所有参数的optimizer

            if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
                return self._get_lr_schedulers([first_stage_optimizer, second_stage_optimizer])

            return [first_stage_optimizer, second_stage_optimizer]

    def training_step(self, batch, batch_ind, optimizer_idx=0):
        logger.info(f'my training_step called in class!!!!!: {self.__class__.__name__}')
        if batch_ind == 0:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f'{key}: {value.shape}--------training_step-------')
                else:
                    logger.debug(f'{key}: type-> {type(value)}')
            logger.debug(f'training_step after print--------')

        for par in self.parameters():
            par.requires_grad = True

        #optimizers = self.optimizers()
        if optimizer_idx == 0:
            treatment_pred = self.get_predictions_first_stage(batch)

            a=batch['current_treatments']
            logger.info(f'treatment_pred shape: {treatment_pred.shape}, aaaa: {a.shape}')
            bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
            bce_loss = self.alpha * bce_loss

            # Masking for shorter sequences
            bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
            logger.info(f'optimizer_idx: {optimizer_idx}, BCE_LOSS:{bce_loss}')

            batch['treatment_pred'] = treatment_pred.detach() #要不要detach？
            #batch['treatment_pred'] = treatment_pred
            self.log(f'{self.model_type}_train_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)

            '''self.manual_backward(bce_loss, retain_graph=True)
            optimizers[0].step()  # 更新优化器
            optimizers[0].zero_grad()  # 清空梯度'''
            return bce_loss
            #return {'loss': bce_loss, 'retain_graph': True}

        elif optimizer_idx == 1:
            treatment_pred = batch['treatment_pred']
            outcome_pred, _ = self.get_predictions_second_stage(batch, treatment_pred)

            mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
            mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()
            logger.info(f'optimizer_idx: {optimizer_idx}, MSE_LOSS:{mse_loss}')
            self.log(f'{self.model_type}_train_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)

            '''self.manual_backward(mse_loss)
            optimizers[1].step()  
            optimizers[1].zero_grad()  '''
            return mse_loss


    '''def training_step(self, batch, batch_ind, optimizer_idx=0):
        self.ema_treatment.to(self.device)
        self.ema_non_treatment.to(self.device)
        # Step 1: Train A (first stage, treatment prediction)
        treatment_pred = self.get_predictions_first_stage(batch)

        # Calculate BCE loss for treatment prediction (A network)
        bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
        bce_loss = self.alpha * bce_loss

        # Masking for shorter sequences
        bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
        logger.info(f'BCE_LOSS: {bce_loss}')

        # Step 2: Train B (second stage, outcome prediction)
        outcome_pred, _ = self.get_predictions_second_stage(batch, treatment_pred)

        # Calculate MSE loss for outcome prediction (B network)
        mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
        mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()
        logger.info(f'MSE_LOSS: {mse_loss}')

        # Joint loss: combine the losses from both stages
        total_loss = bce_loss + mse_loss
        logger.info(f'Total Loss: {total_loss}')

        # Logging the individual losses for monitoring
        self.log(f'{self.model_type}_train_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_train_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)

        # Return the combined loss
        return total_loss'''

    '''def training_step(self, batch, batch_idx, optimizer_idx=0):
        self.ema_treatment.to(self.device)
        self.ema_non_treatment.to(self.device)
        # 如果当前 epoch 早于设定的联合训练 epoch，则只优化第一阶段
        if self.current_epoch < 15:
            treatment_pred = self.get_predictions_first_stage(batch)
            bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
            bce_loss = self.alpha * bce_loss
            bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
            self.log(f'{self.model_type}_train_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
            logger.info(f'epoch: {self.current_epoch}, batch_idx: {batch_idx}--------------------------')
            logger.info(f'treatment_pred:{treatment_pred}')
            a=batch['current_treatments']
            logger.info(f'current_treatments:{a}')
            logger.info(f'diff:{a-treatment_pred}')
            return bce_loss
        else:
            # 联合训练两个阶段
            treatment_pred = self.get_predictions_first_stage(batch)
            bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
            bce_loss = self.alpha * bce_loss
            bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()

            treatment_pred = treatment_pred.detach()

            outcome_pred, _ = self.get_predictions_second_stage(batch, treatment_pred)
            mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
            mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()

            total_loss = bce_loss + mse_loss
            self.log(f'{self.model_type}_train_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
            return total_loss'''

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        logger.debug(f'my predict_step called in class: {self.__class__.__name__}')
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logger.debug(f'{key}: {value.shape}--------predict_step-------')
            else:
                logger.debug(f'{key}: type-> {type(value)}')
        logger.debug(f'predict_step after print--------')

        """
        Generates normalised output predictions
        """
        #treatment_pred = self.get_predictions_first_stage(batch)
        treatment_real = batch['current_treatments']

        '''if self.hparams.exp.weights_ema:
            with self.ema_non_treatment.average_parameters():
                outcome_pred, br = self.get_predictions_second_stage(batch, treatment_real)
        else:
            outcome_pred, br = self.get_predictions_second_stage(batch, treatment_real)'''
        outcome_pred, br = self.get_predictions_second_stage(batch, treatment_real)
        logger.debug(f'my predict_step return: outcome_pred shape :{outcome_pred.shape}')
        return outcome_pred.cpu(), br

    def test_step(self, batch, batch_idx):
        logger.debug(f'my test_step called in class: {self.__class__.__name__}')
        treatment_pred = self.get_predictions_first_stage(batch)

        '''if self.hparams.exp.weights_ema:
            with self.ema_non_treatment.average_parameters():
                with self.ema_treatment.average_parameters():
                    outcome_pred, _ = self.get_predictions_second_stage(batch, treatment_pred)
        else:
            outcome_pred, _ = self.get_predictions_second_stage(batch, treatment_pred)'''
        outcome_pred, _ = self.get_predictions_second_stage(batch, treatment_pred)

        bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
        bce_loss = self.alpha * bce_loss
        mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)

        bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
        mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()
        loss = bce_loss + mse_loss

        subset_name = self.test_dataloader().dataset.subset_name
        self.log(f'{self.model_type}_{subset_name}_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_{subset_name}_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_{subset_name}_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
