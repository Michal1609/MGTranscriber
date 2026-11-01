"""The part of `asteroid <https://github.com/asteroid-team/asteroid>`_ 0.5.2 that pyannote.audio's ToTaToNet
(the network of the PixIT speech separation pipeline) imports: ``DPRNN`` and ``pad_x_to_y``.

Vendored because the ``asteroid`` package pins an old pytorch-lightning, which would downgrade the one pyannote.audio
4 needs. Module and parameter names are kept exactly, so the published checkpoints load. Only the default
configuration of the network is supported (no mulcat RNN, no complex layers, no TorchScript tracing helpers).
``separation.install_asteroid_shim`` registers this module as ``asteroid`` only when the real package is missing.

Adapted from asteroid 0.5.2, Copyright (c) 2019 Pariente Manuel, MIT License:

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without
    limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
    Software, and to permit persons to whom the Software is furnished to do so, subject to the following
    conditions: The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Reference: Y. Luo, Z. Chen, T. Yoshioka, "Dual-path RNN: efficient long sequence modeling for time-domain
single-channel speech separation", https://arxiv.org/abs/1910.06379
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import fold, unfold

_NORM_EPS = 1e-8


class _LayerNorm(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """Input of size ``[batch, channel, *]``."""
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global layer normalization (gLN)."""

    def forward(self, x, eps: float = _NORM_EPS):
        dims = list(range(1, x.dim()))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.var(x, dim=dims, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / torch.sqrt(var + eps))


class ChanLN(_LayerNorm):
    """Channel-wise layer normalization (cLN)."""

    def forward(self, x, eps: float = _NORM_EPS):
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / (var + eps).sqrt())


class CumLN(_LayerNorm):
    """Cumulative global layer normalization (cgLN)."""

    def forward(self, x, eps: float = _NORM_EPS):
        _, chan, spec_len = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=-1)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=-1)
        cnt = torch.arange(start=chan, end=chan * (spec_len + 1), step=chan, dtype=x.dtype, device=x.device).view(1, 1, -1)
        cum_mean = cum_sum / cnt
        cum_var = cum_pow_sum - cum_mean.pow(2)
        return self.apply_gain_and_bias((x - cum_mean) / (cum_var + eps).sqrt())


_NORMS = {"gLN": GlobLN, "cLN": ChanLN, "cgLN": CumLN}
_ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "prelu": nn.PReLU, "gelu": nn.GELU,
                "linear": nn.Identity, "softmax": nn.Softmax}


class SingleRNN(nn.Module):
    """A batch-first RNN/LSTM/GRU layer stack; ``rnn`` holds the parameters."""

    def __init__(self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0, bidirectional=False):
        super().__init__()
        rnn_type = rnn_type.upper()
        if rnn_type not in ("RNN", "LSTM", "GRU"):
            raise ValueError(f"unknown rnn_type {rnn_type}")
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers=n_layers, dropout=dropout,
                                         batch_first=True, bidirectional=bool(bidirectional))

    @property
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, inp):
        """``[batch, seq, feats]``."""
        self.rnn.flatten_parameters()
        out, _ = self.rnn(inp)
        return out


class DPRNNBlock(nn.Module):
    """Dual-path RNN block: an intra-chunk bidirectional RNN followed by an inter-chunk RNN, both residual."""

    def __init__(self, in_chan, hid_size, norm_type="gLN", bidirectional=True, rnn_type="LSTM", use_mulcat=False,
                 num_layers=1, dropout=0):
        super().__init__()
        if use_mulcat:
            raise NotImplementedError("mulcat RNNs are not part of the vendored asteroid subset")
        self.intra_RNN = SingleRNN(rnn_type, in_chan, hid_size, num_layers, dropout=dropout, bidirectional=True)
        self.inter_RNN = SingleRNN(rnn_type, in_chan, hid_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.intra_linear = nn.Linear(self.intra_RNN.output_size, in_chan)
        self.intra_norm = _NORMS[norm_type](in_chan)
        self.inter_linear = nn.Linear(self.inter_RNN.output_size, in_chan)
        self.inter_norm = _NORMS[norm_type](in_chan)

    def forward(self, x):
        """``[batch, feats, chunk_size, num_chunks]``."""
        B, N, K, L = x.size()
        output = x
        x = x.transpose(1, -1).reshape(B * L, K, N)
        x = self.intra_RNN(x)
        x = self.intra_linear(x)
        x = x.reshape(B, L, K, N).transpose(1, -1)
        x = self.intra_norm(x)
        output = output + x
        x = output.transpose(1, 2).transpose(2, -1).reshape(B * K, L, N)
        x = self.inter_RNN(x)
        x = self.inter_linear(x)
        x = x.reshape(B, K, L, N).transpose(1, -1).transpose(2, -1).contiguous()
        x = self.inter_norm(x)
        return output + x


class DPRNN(nn.Module):
    """Dual-path RNN masking network: mask estimator for ``n_src`` sources from ``[batch, in_chan, frames]``."""

    def __init__(self, in_chan, n_src, out_chan=None, bn_chan=128, hid_size=128, chunk_size=100, hop_size=None,
                 n_repeats=6, norm_type="gLN", mask_act="relu", bidirectional=True, rnn_type="LSTM", use_mulcat=False,
                 num_layers=1, dropout=0):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan if out_chan is not None else in_chan
        self.bn_chan = bn_chan
        self.chunk_size = chunk_size
        self.hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.n_src = n_src

        self.bottleneck = nn.Sequential(_NORMS[norm_type](in_chan), nn.Conv1d(in_chan, bn_chan, 1))
        self.net = nn.Sequential(*[
            DPRNNBlock(bn_chan, hid_size, norm_type=norm_type, bidirectional=bidirectional, rnn_type=rnn_type,
                       use_mulcat=use_mulcat, num_layers=num_layers, dropout=dropout)
            for _ in range(n_repeats)])
        self.first_out = nn.Sequential(nn.PReLU(), nn.Conv2d(bn_chan, n_src * bn_chan, 1))
        self.net_out = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bn_chan, self.out_chan, 1, bias=False)
        activation = _ACTIVATIONS[mask_act]
        self.output_act = activation(dim=1) if mask_act == "softmax" else activation()

    def forward(self, mixture_w):
        """Returns the masks, ``[batch, n_src, out_chan, frames]``."""
        batch, _, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        output = unfold(output.unsqueeze(-1), kernel_size=(self.chunk_size, 1), padding=(self.chunk_size, 0),
                        stride=(self.hop_size, 1))
        n_chunks = output.shape[-1]
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        output = self.net(output)
        output = self.first_out(output)
        output = output.reshape(batch * self.n_src, self.bn_chan, self.chunk_size, n_chunks)
        to_unfold = self.bn_chan * self.chunk_size
        output = fold(output.reshape(batch * self.n_src, to_unfold, n_chunks), (n_frames, 1),
                      kernel_size=(self.chunk_size, 1), padding=(self.chunk_size, 0), stride=(self.hop_size, 1))
        output = output.reshape(batch * self.n_src, self.bn_chan, -1)
        output = self.net_out(output) * self.net_gate(output)
        est_mask = self.output_act(self.mask_net(output))
        return est_mask.view(batch, self.n_src, self.out_chan, n_frames)


def pad_x_to_y(x: torch.Tensor, y: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Right-pad or right-trim ``x`` to the size of ``y`` along the last axis."""
    if axis != -1:
        raise NotImplementedError
    return nn.functional.pad(x, [0, y.shape[axis] - x.shape[axis]])
