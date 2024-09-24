import torch
from torch import nn
import math
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def get_fixed_sin_cos_encodings(d_model, max_len):
    """
    Sin-cos fixed positional encodddings
    Args:
        d_model: hidden state dimensionality
        max_len: max sequence length
    Returns: PE
    """
    assert d_model % 2 == 0
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int, trainable=False):
        super().__init__()
        self.max_len = max_len
        self.trainable = trainable
        if trainable:
            self.pe = nn.Embedding(max_len, d_model)
        else:
            self.register_buffer('pe', get_fixed_sin_cos_encodings(d_model, max_len))

    def forward(self, x):
        batch_size = x.size(0)
        actual_len = x.shape[1]
        logger.debug(f'actual_len: {actual_len},,,,,,,,,,,,,,,,,,,,,,,,,,,,,,self.max_len:{self.max_len}')
        assert actual_len <= self.max_len

        pe = self.pe.weight if self.trainable else self.pe
        return pe.unsqueeze(0).repeat(batch_size, 1, 1)[:, :actual_len, :]

    def get_pe(self, position):
        pe = self.pe.weight if self.trainable else self.pe
        return pe[position]


class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_relative_position: int, d_model: int, trainable=False, cross_attn=False):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.trainable = trainable
        self.cross_attn = cross_attn
        self.num_embeddings = (max_relative_position * 2 + 1) if not cross_attn else (max_relative_position + 1)
        if trainable:
            self.embeddings_table = nn.Embedding(self.num_embeddings, d_model)
        else:
            self.register_buffer('embeddings_table', get_fixed_sin_cos_encodings(d_model, max_relative_position * 2 + 1))

    def forward(self, length_q, length_k):
        embeddings_table = self.embeddings_table.weight if self.trainable else self.embeddings_table

        if self.cross_attn:
            distance_mat = torch.arange(length_k - 1, -1, -1)[None, :] + torch.arange(length_q)[:, None]
        else:
            distance_mat = torch.arange(length_k)[None, :] - torch.arange(length_q)[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        if not self.cross_attn:
            distance_mat_clipped = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(distance_mat_clipped)
        embeddings = embeddings_table[final_mat]

        # Non-trainable far-away encodings
        # embeddings[(final_mat == final_mat.min()) | (final_mat == final_mat.max())] = 0.0
        return embeddings


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class Attention(nn.Module):
    def __init__(self,
                 positional_encoding_k: RelativePositionalEncoding = None,
                 positional_encoding_v: RelativePositionalEncoding = None):
        super(Attention, self).__init__()
        self.positional_encoding_k = positional_encoding_k
        self.positional_encoding_v = positional_encoding_v

    def forward(self, query, key, value, mask=None, dropout=None, one_direction=False):
        logger.debug(f'attention  query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}-----------')
        scores = torch.matmul(query, key.transpose(-2, -1))

        if self.positional_encoding_k is not None:
            R_k = self.positional_encoding_k(query.size(2), key.size(2))
            scores = scores + torch.einsum('b h q d, q k d -> b h q k', query, R_k)

        #query,key,value shape都是[488,2,59,8]
        #p_attn,scores shape都是[488,2,59,59]
        scores = scores / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        if one_direction:  # Required for self-attention, but not for cross-attention
            direction_mask = torch.ones_like(scores)
            direction_mask = torch.tril(direction_mask)
            scores = scores.masked_fill(direction_mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        output = torch.matmul(p_attn, value)
        #此处output shape是[488,2,59,8]

        if self.positional_encoding_v is not None:
            R_v = self.positional_encoding_v(query.size(2), value.size(2))
            output = output + torch.einsum('b h q v, q v d -> b h q d', p_attn, R_v)

        logger.debug(f'scoressss shape: {scores.shape}')
        logger.debug(f'maskkkkkk shape: {mask.shape}')
        logger.debug(f'outputttt shape: {output.shape}')
        logger.debug(f'p_attnnnn shape: {p_attn.shape}')
        #R_k以及R_v shape都是[59,59,8]
        #此处output shape还是[488,2,59,8]
        return output, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, head_size=None, dropout=0.0, positional_encoding_k=None, positional_encoding_v=None,
                 #         2         4         4
                 final_layer=False):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        if head_size is not None:
            self.head_size = head_size
        else:
            self.head_size = d_model // num_heads

        logger.info(f'd_model :{d_model}, num_heads :{num_heads}, head_size :{head_size}--->self.head_size: {self.head_size}?????????????????? ')
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, self.num_heads * self.head_size) for _ in range(3)])
                                            #           4                2*4
        self.attention = Attention(positional_encoding_k, positional_encoding_v)
        self.dropout = nn.Dropout(p=dropout)
        if final_layer:
            self.final_layer = nn.Linear(self.num_heads * self.head_size, d_model)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, query, key, value, mask=None, one_direction=True):
        batch_size = query.size(0)

        logger.debug(f'HHquery shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}')
        # 1) do all the linear projections in batch from d_model => num_heads x d_k
        query_, key_, value_ = [layer(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
                                for layer, x in zip(self.linear_layers, (query, key, value))]
        # 2) apply self_attention on all the projected vectors in batch.
        logger.debug(f'HHHHquery_ shape: {query_.shape}, key_ shape: {key_.shape}, value_ shape: {value_.shape}')
        x, attn = self.attention(query_, key_, value_, mask=mask, dropout=self.dropout, one_direction=one_direction)
        logger.debug(f'IIIIx shape: {x.shape}, attn shape: {attn.shape}')

        # 3) "concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_size)
        if hasattr(self, 'final_layer'):
            x = self.final_layer(x)

        return self.layer_norm(x + query)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
        return self.layer_norm(self.dropout(self.conv2(x_)).permute(0, 2, 1) + x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden, attn_heads, head_size, feed_forward_hidden, dropout, attn_dropout=0.1,
                 self_positional_encoding_k=None, self_positional_encoding_v=None, final_layer=True, **kwargs):
        super().__init__()
        # self.layer_norm = LayerNorm(hidden)   - already used in MultiHeadedAttention and PositionwiseFeedForward
        self.self_attention = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                   #            2                  4                   4
                                                   dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                   positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        #                                                     4                   16

    def forward(self, x, active_entries):
        logger.debug(f'active_entries shape: {active_entries.shape}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self_att_mask = active_entries.repeat(1, 1, active_entries.size(1)).unsqueeze(1)
        logger.debug(f'xxxxxxxxx shape: {x.shape}')
        logger.debug(f'active_entries shape: {active_entries.shape}')
        logger.debug(f'self_att_mask shape: {self_att_mask.shape}')
        x = self.self_attention(x, x, x, self_att_mask, True)
        x = self.feed_forward(x)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden, attn_heads, head_size, feed_forward_hidden, dropout, attn_dropout,
                 self_positional_encoding_k=None, self_positional_encoding_v=None,
                 cross_positional_encoding_k=None, cross_positional_encoding_v=None, final_layer=False, **kwargs):
        super().__init__()
        self.layer_norm = LayerNorm(hidden)
        self.self_attention = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                   dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                   positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)
        self.cross_attention = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                    dropout=attn_dropout, positional_encoding_k=cross_positional_encoding_k,
                                                    positional_encoding_v=cross_positional_encoding_v, final_layer=final_layer)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)

    def forward(self, x, encoder_x, active_entries, active_encoder_br):
        logger.debug(f'x shape:{x.shape},  encoder_x shape:{encoder_x.shape}, active_entries shape:{active_entries}, active_encoder_br shape:{active_encoder_br}')
        self_att_mask = active_entries.repeat(1, 1, active_entries.size(1)).unsqueeze(1)
        cross_att_mask = (active_encoder_br.unsqueeze(1) * active_entries).unsqueeze(1)
        logger.debug(f'self_att_mask:{self_att_mask}, cross_att_mask:{cross_att_mask}')
        logger.debug(f'x1 shape:{x.shape}')
        x = self.self_attention(x, x, x, self_att_mask, True)
        logger.debug(f'x2 shape:{x.shape}')
        x = self.cross_attention(x, encoder_x, encoder_x, cross_att_mask, False)
        logger.debug(f'x3 shape:{x.shape}')
        x = self.feed_forward(x)
        logger.debug(f'x4 shape:{x.shape}')
        return x


class TransformerMultiInputBlock(nn.Module):
    def __init__(self, hidden, attn_heads, head_size, feed_forward_hidden, dropout, attn_dropout,
                 self_positional_encoding_k=None, self_positional_encoding_v=None, n_inputs=2, final_layer=False,
                 disable_cross_attention=False, isolate_subnetwork='', **kwargs):
        super().__init__()
        self.n_inputs = n_inputs
        self.disable_cross_attention = disable_cross_attention
        self.isolate_subnetwork = isolate_subnetwork
        self.self_attention_o = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                     dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                     positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)
        self.self_attention_t = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                     dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                     positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)
        if not disable_cross_attention:
            self.cross_attention_ot = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                           dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                           positional_encoding_v=self_positional_encoding_v,
                                                           final_layer=final_layer)
            self.cross_attention_to = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                           dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                           positional_encoding_v=self_positional_encoding_v,
                                                           final_layer=final_layer)

        if n_inputs == 3:
            self.self_attention_v = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                         dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                         positional_encoding_v=self_positional_encoding_v, final_layer=final_layer
                                                         )
            if not disable_cross_attention:
                self.cross_attention_tv = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                               dropout=attn_dropout,
                                                               positional_encoding_k=self_positional_encoding_k,
                                                               positional_encoding_v=self_positional_encoding_v,
                                                               final_layer=final_layer)
                self.cross_attention_vt = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                               dropout=attn_dropout,
                                                               positional_encoding_k=self_positional_encoding_k,
                                                               positional_encoding_v=self_positional_encoding_v,
                                                               final_layer=final_layer)
                self.cross_attention_ov = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                               dropout=attn_dropout,
                                                               positional_encoding_k=self_positional_encoding_k,
                                                               positional_encoding_v=self_positional_encoding_v,
                                                               final_layer=final_layer)
                self.cross_attention_vo = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                               dropout=attn_dropout,
                                                               positional_encoding_k=self_positional_encoding_k,
                                                               positional_encoding_v=self_positional_encoding_v,
                                                               final_layer=final_layer)

        self.feed_forwards = [PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
                              for _ in range(n_inputs)]
        self.feed_forwards = nn.ModuleList(self.feed_forwards)

        self.n_inputs = n_inputs

    def forward(self, x_tov, x_s, active_entries_treat_outcomes, active_entries_vitals=None):
        assert len(x_tov) == self.n_inputs
        if self.n_inputs == 2:
            x_t, x_o = x_tov
        else:
            x_t, x_o, x_v = x_tov

        logger.debug(f'active_entries_treat_outcomes shape~~~~~~: {active_entries_treat_outcomes.shape}')
        self_att_mask_ot = active_entries_treat_outcomes.repeat(1, 1, x_t.size(1)).unsqueeze(1)
        logger.debug(f'self_att_mask_ot shape~~~~~: {self_att_mask_ot.shape}')
        cross_att_mask_ot = cross_att_mask_to = self_att_mask_ot

        x_t_ = self.self_attention_t(x_t, x_t, x_t, self_att_mask_ot, True)
        x_to_ = self.cross_attention_to(x_t_, x_o, x_o, cross_att_mask_ot, True) if not self.disable_cross_attention \
            and self.isolate_subnetwork != 't' and self.isolate_subnetwork != 'o' else x_t_

        x_o_ = self.self_attention_o(x_o, x_o, x_o, self_att_mask_ot, True)
        x_ot_ = self.cross_attention_ot(x_o_, x_t, x_t, cross_att_mask_to, True) if not self.disable_cross_attention \
            and self.isolate_subnetwork != 'o' and self.isolate_subnetwork != 't' else x_o_

        if self.n_inputs == 2:
            out_t = self.feed_forwards[0](x_to_ + x_s)
            out_o = self.feed_forwards[1](x_ot_ + x_s)

            return out_t, out_o

        else:
            self_att_mask_v = active_entries_vitals.repeat(1, 1, x_v.size(1)).unsqueeze(1)
            cross_att_mask_ot_v = (active_entries_vitals.squeeze(-1).unsqueeze(1) * active_entries_treat_outcomes).unsqueeze(1)
            cross_att_mask_v_ot = (active_entries_treat_outcomes.squeeze(-1).unsqueeze(1) * active_entries_vitals).unsqueeze(1)

            x_tv_ = self.cross_attention_tv(x_t_, x_v, x_v, cross_att_mask_ot_v, True) if not self.disable_cross_attention \
                and self.isolate_subnetwork != 't' and self.isolate_subnetwork != 'v' else 0.0
            x_ov_ = self.cross_attention_ov(x_o_, x_v, x_v, cross_att_mask_ot_v, True) if not self.disable_cross_attention \
                and self.isolate_subnetwork != 'o' and self.isolate_subnetwork != 'v' else 0.0

            x_v_ = self.self_attention_v(x_v, x_v, x_v, self_att_mask_v, True)
            x_vt_ = self.cross_attention_vt(x_v_, x_t, x_t, cross_att_mask_v_ot, True) if not self.disable_cross_attention \
                and self.isolate_subnetwork != 'v' and self.isolate_subnetwork != 't' else x_v_
            x_vo_ = self.cross_attention_vo(x_v_, x_o, x_o, cross_att_mask_v_ot, True) if not self.disable_cross_attention \
                and self.isolate_subnetwork != 'v' and self.isolate_subnetwork != 'o' else 0.0

            out_t = self.feed_forwards[0](x_to_ + x_tv_ + x_s)
            out_o = self.feed_forwards[1](x_ot_ + x_ov_ + x_s)
            out_v = self.feed_forwards[2](x_vt_ + x_vo_ + x_s)

            return out_t, out_o, out_v


class IVTransformerMultiInputBlock(nn.Module):
    def __init__(self, hidden, attn_heads, head_size, feed_forward_hidden, dropout, attn_dropout,
                 self_positional_encoding_k=None, self_positional_encoding_v=None, n_inputs=4, final_layer=False,
                 disable_cross_attention=False, isolate_subnetwork='', **kwargs):
        super().__init__()
        self.n_inputs = n_inputs
        self.disable_cross_attention = disable_cross_attention
        self.isolate_subnetwork = isolate_subnetwork
        self.self_attention_chemo_iv = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                     dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                     positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)
        self.self_attention_radio_iv = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                     dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                     positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)

        self.self_attention_o = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                     dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                     positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)
        self.self_attention_t = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                     dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                     positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)
        if not disable_cross_attention:
            self.cross_attention_ot = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                           dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                           positional_encoding_v=self_positional_encoding_v,
                                                           final_layer=final_layer)
            self.cross_attention_to = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                           dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                           positional_encoding_v=self_positional_encoding_v,
                                                           final_layer=final_layer)

        if n_inputs == 5:
            self.self_attention_v = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                         dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                         positional_encoding_v=self_positional_encoding_v, final_layer=final_layer
                                                         )
            if not disable_cross_attention:
                self.cross_attention_tv = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                               dropout=attn_dropout,
                                                               positional_encoding_k=self_positional_encoding_k,
                                                               positional_encoding_v=self_positional_encoding_v,
                                                               final_layer=final_layer)
                self.cross_attention_vt = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                               dropout=attn_dropout,
                                                               positional_encoding_k=self_positional_encoding_k,
                                                               positional_encoding_v=self_positional_encoding_v,
                                                               final_layer=final_layer)
                self.cross_attention_ov = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                               dropout=attn_dropout,
                                                               positional_encoding_k=self_positional_encoding_k,
                                                               positional_encoding_v=self_positional_encoding_v,
                                                               final_layer=final_layer)
                self.cross_attention_vo = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                               dropout=attn_dropout,
                                                               positional_encoding_k=self_positional_encoding_k,
                                                               positional_encoding_v=self_positional_encoding_v,
                                                               final_layer=final_layer)

        logger.debug(f'hidden size: {hidden}, feed_forward_hidden_size: {feed_forward_hidden}uuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')
        self.feed_forwards = [PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
                              for _ in range(n_inputs)]
        self.feed_forwards = nn.ModuleList(self.feed_forwards)

        self.n_inputs = n_inputs

    def forward(self, x_tov, x_s, active_entries_treat_outcomes, active_entries_vitals=None):
        assert len(x_tov) == self.n_inputs
        if self.n_inputs == 4:
            x_t, x_o, x_chemo_iv, x_radio_iv = x_tov
            #logger.debug(f'AAAx_t shape: {x_t.shape}, x_o shape: {x_o.shape}, x_chemo_iv shape: {x_chemo_iv.shape}, x_radio_iv shape: {x_radio_iv.shape}')
        else:
            x_t, x_o, x_chemo_iv, x_radio_iv, x_v = x_tov
            #logger.debug(f'AAAAx_t shape: {x_t.shape}, x_o shape: {x_o.shape}, x_v shape: {x_v.shape}, x_chemo_iv shape: {x_chemo_iv.shape}, x_radio_iv shape: {x_radio_iv.shape}')

        self_att_mask_ot = active_entries_treat_outcomes.repeat(1, 1, x_t.size(1)).unsqueeze(1)
        cross_att_mask_ot = cross_att_mask_to = self_att_mask_ot
        self_att_mask_chemoiv_radioiv = self_att_mask_ot

        logger.debug(f'active_entries_treat_outcomes shape: {active_entries_treat_outcomes.shape}######')
        logger.debug(f'self_att_mask_ot shape: {self_att_mask_ot.shape}')

        x_chemo_iv_ = self.self_attention_chemo_iv(x_chemo_iv, x_chemo_iv, x_chemo_iv, self_att_mask_chemoiv_radioiv, True) if x_chemo_iv is not None else None
        x_radio_iv_ = self.self_attention_radio_iv(x_radio_iv, x_radio_iv, x_radio_iv, self_att_mask_chemoiv_radioiv, True) if x_radio_iv is not None else None
        #logger.debug(f'BBBx_chemo_iv_ shape: {x_chemo_iv_.shape}, x_radio_iv_ shape: {x_radio_iv_.shape}')

        x_t_ = self.self_attention_t(x_t, x_t, x_t, self_att_mask_ot, True)
        logger.debug(f'CCCCx_t_ shape: {x_t_.shape}')
        x_to_ = self.cross_attention_to(x_t_, x_o, x_o, cross_att_mask_ot, True) if not self.disable_cross_attention \
            and self.isolate_subnetwork != 't' and self.isolate_subnetwork != 'o' else x_t_
        logger.debug(f'DDDDx_to_ shape: {x_to_.shape}')

        x_o_ = self.self_attention_o(x_o, x_o, x_o, self_att_mask_ot, True)
        logger.debug(f'EEEEx_o_ shape: {x_o_.shape}')
        x_ot_ = self.cross_attention_ot(x_o_, x_t, x_t, cross_att_mask_to, True) if not self.disable_cross_attention \
            and self.isolate_subnetwork != 'o' and self.isolate_subnetwork != 't' else x_o_
        logger.debug(f'FFFFx_ot_ shape: {x_ot_.shape}')

        if self.n_inputs == 4:
            out_t = self.feed_forwards[0](x_to_ + x_s)
            out_o = self.feed_forwards[1](x_ot_ + x_s)
            out_chemo_iv = self.feed_forwards[2](x_chemo_iv_) if x_chemo_iv is not None else None
            out_radio_iv = self.feed_forwards[3](x_radio_iv_) if x_radio_iv is not None else None

            #logger.debug(f'GGGG1out_t shape: {out_t.shape}, out_o shape: {out_o.shape}, out_chemo_iv: {out_chemo_iv.shape}, out_radio_iv: {out_radio_iv.shape}')
            return out_t, out_o, out_chemo_iv, out_radio_iv

        else:
            self_att_mask_v = active_entries_vitals.repeat(1, 1, x_v.size(1)).unsqueeze(1)
            cross_att_mask_ot_v = (active_entries_vitals.squeeze(-1).unsqueeze(1) * active_entries_treat_outcomes).unsqueeze(1)
            cross_att_mask_v_ot = (active_entries_treat_outcomes.squeeze(-1).unsqueeze(1) * active_entries_vitals).unsqueeze(1)

            x_tv_ = self.cross_attention_tv(x_t_, x_v, x_v, cross_att_mask_ot_v, True) if not self.disable_cross_attention \
                and self.isolate_subnetwork != 't' and self.isolate_subnetwork != 'v' else 0.0
            x_ov_ = self.cross_attention_ov(x_o_, x_v, x_v, cross_att_mask_ot_v, True) if not self.disable_cross_attention \
                and self.isolate_subnetwork != 'o' and self.isolate_subnetwork != 'v' else 0.0

            x_v_ = self.self_attention_v(x_v, x_v, x_v, self_att_mask_v, True)
            x_vt_ = self.cross_attention_vt(x_v_, x_t, x_t, cross_att_mask_v_ot, True) if not self.disable_cross_attention \
                and self.isolate_subnetwork != 'v' and self.isolate_subnetwork != 't' else x_v_
            x_vo_ = self.cross_attention_vo(x_v_, x_o, x_o, cross_att_mask_v_ot, True) if not self.disable_cross_attention \
                and self.isolate_subnetwork != 'v' and self.isolate_subnetwork != 'o' else 0.0

            out_t = self.feed_forwards[0](x_to_ + x_tv_ + x_s)
            out_o = self.feed_forwards[1](x_ot_ + x_ov_ + x_s)
            out_chemo_iv = self.feed_forwards[2](x_chemo_iv_) if x_chemo_iv is not None else None
            out_radio_iv = self.feed_forwards[3](x_radio_iv_) if x_radio_iv is not None else None
            out_v = self.feed_forwards[4](x_vt_ + x_vo_ + x_s)

            #logger.debug(f'GGGG2out_t shape: {out_t.shape}, out_o shape: {out_o.shape}, out_chemo_iv: {out_chemo_iv.shape}, out_radio_iv: {out_radio_iv.shape}, out_v: {out_v.shape}')
            return out_t, out_o, out_chemo_iv, out_radio_iv, out_v

class OnlyIVTransformerMultiInputBlock(nn.Module):
    def __init__(self, hidden, attn_heads, head_size, feed_forward_hidden, dropout, attn_dropout,
                 self_positional_encoding_k=None, self_positional_encoding_v=None, n_iv_inputs=2, final_layer=False,
                 disable_cross_attention=False, isolate_subnetwork='', **kwargs):
        super().__init__()
        self.n_iv_inputs = n_iv_inputs
        self.disable_cross_attention = disable_cross_attention
        self.isolate_subnetwork = isolate_subnetwork
        self.self_attention_chemo_iv = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                     dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                     positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)
        self.self_attention_radio_iv = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                     dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                     positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)

        self.feed_forwards = [PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
                              for _ in range(n_iv_inputs)]
        self.feed_forwards = nn.ModuleList(self.feed_forwards)
        self.n_iv_inputs = n_iv_inputs

    def forward(self, x_tov, active_entries_treat_outcomes, active_entries_vitals=None):
        #assert len(x_tov) == self.n_inputs
        x_chemo_iv, x_radio_iv = x_tov
            #logger.debug(f'AAAx_t shape: {x_t.shape}, x_o shape: {x_o.shape}, x_chemo_iv shape: {x_chemo_iv.shape}, x_radio_iv shape: {x_radio_iv.shape}')

        self_att_mask_ot = active_entries_treat_outcomes.repeat(1, 1, x_chemo_iv.size(1)).unsqueeze(1)
        cross_att_mask_ot = cross_att_mask_to = self_att_mask_ot
        self_att_mask_chemoiv_radioiv = self_att_mask_ot

        x_chemo_iv_ = self.self_attention_chemo_iv(x_chemo_iv, x_chemo_iv, x_chemo_iv, self_att_mask_chemoiv_radioiv, True) if x_chemo_iv is not None else None
        x_radio_iv_ = self.self_attention_radio_iv(x_radio_iv, x_radio_iv, x_radio_iv, self_att_mask_chemoiv_radioiv, True) if x_radio_iv is not None else None
        #logger.debug(f'BBBx_chemo_iv_ shape: {x_chemo_iv_.shape}, x_radio_iv_ shape: {x_radio_iv_.shape}')

        out_chemo_iv = self.feed_forwards[0](x_chemo_iv_) if x_chemo_iv is not None else None
        out_radio_iv = self.feed_forwards[1](x_radio_iv_) if x_radio_iv is not None else None

            #logger.debug(f'GGGG1out_t shape: {out_t.shape}, out_o shape: {out_o.shape}, out_chemo_iv: {out_chemo_iv.shape}, out_radio_iv: {out_radio_iv.shape}')
        return out_chemo_iv, out_radio_iv