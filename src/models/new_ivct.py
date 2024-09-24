from pytorch_lightning import LightningModule
from omegaconf import DictConfig
import torch
from torch import nn
from omegaconf.errors import MissingMandatoryValue
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset, Subset
import logging
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from functools import partial
import seaborn as sns
from sklearn.manifold import TSNE
import pdb

import pytorch_lightning as pl
from src.models.edct import EDCT
from src.models.utils_transformer import IVTransformerMultiInputBlock, LayerNorm
from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.utils_transformer import TransformerEncoderBlock, TransformerDecoderBlock, AbsolutePositionalEncoding, RelativePositionalEncoding

logger = logging.getLogger(__name__)

class FirstStageModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, head_size, num_layers, max_len):
        #              7            4        2         4
        super(FirstStageModel, self).__init__()
        #self.positional_encoding = RelativePositionalEncoding(max_relative_position=max_len, d_model=d_model)
        #self.fc_in = nn.Linear(input_dim, d_model*num_heads)
        self.fc_in = nn.Linear(input_dim, d_model*num_heads)
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(hidden=d_model*num_heads, attn_heads=num_heads, head_size=head_size, feed_forward_hidden=d_model * 4, dropout=0.1) for _ in range(num_layers)]
            #                                  4                   2                     4                             4*4
        )
        self.fc_out = nn.Linear(d_model*num_heads, input_dim)  # Output: predicted treatments
        #                         4           7
    def forward(self, prev_treatments, static_features, iv_inputs, vitals=None, active_entries=None):
        if vitals is not None:
            x = torch.cat([prev_treatments, vitals, static_features.unsqueeze(1).repeat(1, prev_treatments.size(1), 1), iv_inputs], dim=-1)
        else:
            x = torch.cat([prev_treatments, static_features.unsqueeze(1).repeat(1, prev_treatments.size(1), 1), iv_inputs], dim=-1)
        logger.info(f'FirstStageModel x shape: {x.shape}')
        x = self.fc_in(x)
        #pe = self.positional_encoding(x.size(1), x.size(1))
        #x = x + pe
        #x = self.positional_encoding(x.size(1), x.size(1))
        #logger.info(f'FirstStageModel pe shape: {pe.shape}')
        logger.info(f'FirstStageModel x shape: {x.shape}')
        for block in self.transformer_blocks:
            x = block(x, active_entries)
        x = self.fc_out(x)
        return x


class SecondStageModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, head_size, num_layers, max_len):
        super(SecondStageModel, self).__init__()
        #self.positional_encoding = RelativePositionalEncoding(max_relative_position=max_len, d_model=d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerDecoderBlock(hidden=d_model, attn_heads=num_heads, head_size=head_size, feed_forward_hidden=d_model * 4, dropout=0.1, attn_dropout=0.1) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, 1)  # Output: predicted cancer volume

    def forward(self, prev_outputs, static_features, predicted_treatments, active_entries=None):
        x = torch.cat([prev_outputs, static_features.unsqueeze(1).repeat(1, prev_outputs.size(1), 1), predicted_treatments], dim=-1)
        #pe = self.positional_encoding(x.size(1), x.size(1))
        #x = x + pe
        #x = self.positional_encoding(x.size(1), x.size(1))
        for block in self.transformer_blocks:
            x = block(x, active_entries)
        x = self.fc_out(x)
        return x

class NEWIVCT(EDCT):
    model_type = 'new_iv_multi'
    possible_model_types = ['new_iv_multi']

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] =None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 projection_horizon: int = None,
                 bce_weights: np.array = None, **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        if self.dataset_collection is not None:
            self.projection_horizon = self.dataset_collection.projection_horizon
        else:
            self.projection_horizon = projection_horizon
        #self.dim_iv = self.hparams.model.new_iv_multi.first_stage.dim_iv
        # Used in hparam tuning
        #self.input_size = max(self.dim_treatments, self.dim_static_features, self.dim_vitals, self.dim_outcome)
        logger.info(f'DIM_TREAT:{self.dim_treatments}, DIM_STATIC:{self.dim_static_features}, DIM_VITALS:{self.dim_vitals}, DIM_OUTCOME:{self.dim_outcome}')
        #self.input_dim = self.dim_treatments + self.dim_static_features + self.dim_vitals + self.dim_outcome + 2*self.dim_iv
        #self.input_dim = self.dim_treatments + self.dim_static_features + self.dim_vitals + 2*self.dim_iv
        self.dim_iv = args.model.new_iv_multi.dim_iv
        self.input_dim = self.dim_treatments + self.dim_static_features + self.dim_vitals + 2*self.dim_iv
        logger.info(f'INPUT_DIM:{self.input_dim}')
        logger.info(f'Sum input size of {self.model_type}: {self.input_size}')
        assert self.autoregressive  # prev_outcomes are obligatory

        self.first_stage_model = FirstStageModel(
            self.input_dim,
            self.input_dim,
            #self.hparams.model.new_iv_multi.first_stage.d_model,
            self.hparams.model.new_iv_multi.first_stage.num_heads,
            self.hparams.model.new_iv_multi.first_stage.head_size,
            self.hparams.model.new_iv_multi.first_stage.num_layers,
            self.hparams.dataset.max_seq_length
        )

        self.second_stage_model = SecondStageModel(
            self.dim_outcome + self.dim_treatments,
            self.hparams.model.new_iv_multi.second_stage.d_model,
            self.hparams.model.new_iv_multi.second_stage.num_heads,
            self.hparams.model.new_iv_multi.second_stage.head_size,
            self.hparams.model.new_iv_multi.second_stage.num_layers,
            self.hparams.dataset.max_seq_length
        )

        self.loss_fn_treatment = nn.MSELoss()
        self.loss_fn_outcome = nn.MSELoss()

        #self.new_iv_multi_init_specific(args.model.new_iv_multi)
        self.save_hyperparameters(args)

    def new_iv_multi_init_specific(self, sub_args: DictConfig):
        """
        Initialization of specific sub-network (Encoder/decoder)
        Args:
            sub_args: sub-network hyperparameters
        """
        try:
            self.max_seq_length = sub_args.max_seq_length
            self.br_size = sub_args.br_size  # balanced representation size
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.dropout_rate = sub_args.dropout_rate
            # self.attn_dropout_rate = sub_args.attn_dropout_rate

            self.num_layer = sub_args.num_layer
            self.num_heads = sub_args.num_heads

            if self.seq_hidden_units is None or self.br_size is None or self.fc_hidden_units is None \
                    or self.dropout_rate is None:
                raise MissingMandatoryValue()

            self.head_size = sub_args.seq_hidden_units // sub_args.num_heads

            # Pytorch model init
            self.input_transformation = nn.Linear(self.input_size, self.seq_hidden_units) if self.input_size else None

            # Init of positional encodings
            self.self_positional_encoding = self.self_positional_encoding_k = self.self_positional_encoding_v = None
            if sub_args.self_positional_encoding.absolute:
                self.self_positional_encoding = \
                    AbsolutePositionalEncoding(self.max_seq_length, self.seq_hidden_units,
                                               sub_args.self_positional_encoding.trainable)
            else:
                # Relative positional encoding is shared across heads
                self.self_positional_encoding_k = \
                    RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                               sub_args.self_positional_encoding.trainable)
                self.self_positional_encoding_v = \
                    RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                               sub_args.self_positional_encoding.trainable)

            self.cross_positional_encoding = self.cross_positional_encoding_k = self.cross_positional_encoding_v = None
            if 'cross_positional_encoding' in sub_args and sub_args.cross_positional_encoding.absolute:
                self.cross_positional_encoding = \
                    AbsolutePositionalEncoding(self.max_seq_length, self.seq_hidden_units,
                                               sub_args.cross_positional_encoding.trainable)
            elif 'cross_positional_encoding' in sub_args:
                # Relative positional encoding is shared across heads
                self.cross_positional_encoding_k = \
                    RelativePositionalEncoding(sub_args.cross_positional_encoding.max_relative_position, self.head_size,
                                               sub_args.cross_positional_encoding.trainable, cross_attn=True)
                self.cross_positional_encoding_v = \
                    RelativePositionalEncoding(sub_args.cross_positional_encoding.max_relative_position, self.head_size,
                                               sub_args.cross_positional_encoding.trainable, cross_attn=True)

            self.output_dropout = nn.Dropout(self.dropout_rate)
        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def print_batch_shapes(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"{key}: {value.shape}")
            else:
                logger.info(f"{key}: 非张量类型，类型为 {type(value)}")

    def forward(self, batch):
        iv_inputs = torch.cat([batch['chemo_iv'], batch['radio_iv']], dim=-1)

        self.print_batch_shapes(batch)

        if self.has_vitals:
            predicted_treatments = self.first_stage_model(batch['prev_treatments'], batch['static_features'],
                                                          iv_inputs, batch['vitals'], batch['active_entries'])
        else:
            predicted_treatments = self.first_stage_model(batch['prev_treatments'], batch['static_features'],
                                                          iv_inputs, active_entries=batch['active_entries'])

        logger.info(f'predicted_treatments shape: {predicted_treatments.shape}--------------------------------')
        predicted_outcomes = self.second_stage_model(batch['prev_outputs'], batch['static_features'],
                                                     predicted_treatments, batch['active_entries'])
        return predicted_treatments, predicted_outcomes

    def training_step(self, batch, batch_idx):
        logger.info(f'training_step********************************************************')
        predicted_treatments, predicted_outcomes = self(batch)
        loss_treatment = self.loss_fn_treatment(predicted_treatments, batch['current_treatments'])
        loss_outcome = self.loss_fn_outcome(predicted_outcomes, batch['outputs'])
        total_loss = loss_treatment + loss_outcome
        self.log('train_loss', total_loss)
        self.log('train_loss_treatment', loss_treatment)
        self.log('train_loss_outcome', loss_outcome)
        return total_loss

    def test_step(self, batch, batch_idx):
        logger.info(f'test_step********************************************************')
        predicted_treatments, predicted_outcomes = self(batch)
        loss_treatment = self.loss_fn_treatment(predicted_treatments, batch['current_treatments'])
        loss_outcome = self.loss_fn_outcome(predicted_outcomes, batch['outputs'])
        total_loss = loss_treatment + loss_outcome
        self.log('test_loss', total_loss, on_epoch=True, sync_dist=True)
        self.log('test_loss_treatment', loss_treatment, on_epoch=True, sync_dist=True)
        self.log('test_loss_outcome', loss_outcome, on_epoch=True, sync_dist=True)
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        logger.info(f'ppppppppppppppppprrrrrrrredict step')
        logger.info(f'batch.keys: {batch.keys()}')
        _, predicted_outcomes = self(batch)
        logger.info(f'predicted_outcome shape: {predicted_outcomes.shape}       predicted_outcomes len: {len(predicted_outcomes)} ')
        return predicted_outcomes.cpu()
    def get_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'My MODEL Predictions for {dataset.subset_name}.')
        # Creating Dataloader
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        outcome_pred = torch.cat(self.trainer.predict(self, data_loader), dim=0)
        return outcome_pred.numpy()

    def configure_optimizers(self):
        sub_args = self.hparams.model[self.model_type]
        lr = sub_args['optimizer']['learning_rate']
        return torch.optim.Adam(self.parameters(), lr=lr)
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, *args, **kwargs):
        optimizer.step()
        optimizer.zero_grad()

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Autoregressive Prediction for {dataset.subset_name}.')
        logger.info(f'My Model-----------------------autogressive')

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
