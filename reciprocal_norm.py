import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch
import itertools
from option import args

gpus = args.gpu_id.split(',')
def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        nn.init.constant_(m[-1].weight, val=0)
        nn.init.constant_(m[-1].bias, val=0)
    else:
        nn.init.constant_(m, val=0)

class _ReciprocalNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, is_1d=False):
        super(_ReciprocalNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.group = 1 if num_features < 512 else num_features // 512
        self.my_mean_w_s = nn.Parameter(torch.ones(num_features,1))
        self.my_var_w_s = nn.Parameter(torch.ones(num_features,1))
        self.my_mean_w_t = nn.Parameter(torch.ones(num_features,1))
        self.my_var_w_t = nn.Parameter(torch.ones(num_features,1))

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            self.register_parameter('lamda', None)

        if self.track_running_stats:
            self.register_buffer('running_mean_source', torch.zeros(num_features))
            self.register_buffer('running_mean_target', torch.zeros(num_features))
            self.register_buffer('running_var_source', torch.ones(num_features))
            self.register_buffer('running_var_target', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean_source', None)
            self.register_parameter('running_mean_target', None)
            self.register_parameter('running_var_source', None)
            self.register_parameter('running_var_target', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean_source.zero_()
            self.running_mean_target.zero_()
            self.running_var_source.fill_(1)
            self.running_var_target.fill_(1)
            self.num_batches_tracked.zero_()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def _load_from_state_dict_from_pretrained_model(self, state_dict, prefix, metadata, strict, missing_keys,
                                                    unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr`metadata`.
        For state dicts without meta data, :attr`metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            metadata (dict): a dict containing the metadata for this moodule.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=False``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=False``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if 'source' in key or 'target' in key:
                key = key[:-7]
                # print(key)
            if key in state_dict:
                input_param = state_dict[key]
                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param of {} from checkpoint, '
                                      'where the shape is {} in current model.'
                                      .format(key, param.shape, input_param.shape))
                    continue
                if isinstance(input_param, Parameter):
                    # backwards compatibility for serialized parameters
                    input_param = input_param.data
                try:
                    param.copy_(input_param)
                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)

    def forward(self, input, option='residual', running_flag=False, kernel='Student'):
        self._check_input_dim(input)

        if self.training:  ## train

            ## 1. Domain Specific Mean and Variance.
            if input.dim() == 4:
                b,c,h,w = input.size()
                batch_size = b // 2
            else:
                b,c = input.size()
                batch_size = b // 2
            input_source = input[:batch_size]
            input_target = input[batch_size:]
            source_for_norm = input_source.clone()
            target_for_norm = input_target.clone()
            # print(input.dim())
            if input.dim() == 4:  ## TransNorm2d
                input_source = input_source.permute(1, 0, 2, 3).contiguous().view(self.num_features, -1)  # [c, bhw]
                input_target = input_target.permute(1, 0, 2, 3).contiguous().view(self.num_features, -1)
            else:
                input_source = input_source.permute(1, 0).contiguous().view(self.num_features, -1)  # [c, bhw]
                input_target = input_target.permute(1, 0).contiguous().view(self.num_features, -1)

            cur_mean_source = torch.mean(input_source, dim=1).view(-1, 1)
            cur_var_source = torch.var(input_source, dim=1).view(-1, 1)
            cur_mean_target = torch.mean(input_target, dim=1).view(-1, 1)
            cur_var_target = torch.var(input_target, dim=1).view(-1, 1)
            if self.group > 1:
                cur_mean_source = cur_mean_source.view(c // self.group, self.group)
                cur_var_source = cur_var_source.view(c // self.group, self.group)
                cur_mean_target = cur_mean_target.view(c // self.group, self.group)
                cur_var_target = cur_var_target.view(c // self.group, self.group)


            if args.dist == 'l1':
                mean_dis_st = -1 * torch.abs(cur_mean_source - cur_mean_target.permute(1, 0).contiguous())
                mean_dis_ts = -1 * torch.abs(cur_mean_target - cur_mean_source.permute(1, 0).contiguous())
                var_dis_st = -1 * torch.abs(cur_var_source - cur_var_target.permute(1, 0).contiguous())
                var_dis_ts = -1 * torch.abs(cur_var_target - cur_var_source.permute(1, 0).contiguous())

            elif args.dist == 'l2':
                ## l2
                if self.group > 1:
                    mean_s_l2, mean_t_l2, var_s_l2, var_t_l2 = cur_mean_source.mean(1, keepdim=True), cur_mean_target.mean(1, keepdim=True), cur_var_source.mean(1,keepdim=True), cur_var_target.mean(1,keepdim=True)
                else:
                    mean_s_l2, mean_t_l2, var_s_l2, var_t_l2 = cur_mean_source, cur_mean_target, cur_var_source, cur_var_target

                mean_dis_st = -1 * torch.pow(mean_s_l2 - mean_t_l2.permute(1, 0).contiguous(), 2)  #[c,1] - [1,c]
                mean_dis_ts = -1 * torch.pow(mean_t_l2 - mean_s_l2.permute(1, 0).contiguous(), 2)  # [c,c]
                var_dis_st = -1 * torch.pow(var_s_l2 - var_t_l2.permute(1, 0).contiguous(), 2)
                var_dis_ts = -1 * torch.pow(var_t_l2 - var_s_l2.permute(1, 0).contiguous(), 2)

            elif args.dist == 'cosine':
                # cosine
                mean_dis_st = torch.matmul(cur_mean_source, cur_mean_target.t())
                mean_dis_ts = torch.matmul(cur_mean_target, cur_mean_source.t())
                var_dis_st = torch.matmul(cur_var_source, cur_var_target.t())
                var_dis_ts = torch.matmul(cur_var_target, cur_var_source.t())

            mean_pro_st = F.softmax(mean_dis_st,dim=1)  #
            mean_pro_ts = F.softmax(mean_dis_ts,dim=1)
            var_pro_st = F.softmax(var_dis_st, dim=1)  #
            var_pro_ts = F.softmax(var_dis_ts, dim=1)

            mean_s_in_t = torch.matmul(mean_pro_st, cur_mean_target)  # [c//g,g]
            mean_t_in_s = torch.matmul(mean_pro_ts, cur_mean_source)
            var_s_in_t = torch.matmul(var_pro_st, cur_var_target)
            var_t_in_s = torch.matmul(var_pro_ts, cur_var_source)

            if self.group > 1:
                mean_s_in_t = mean_s_in_t.view(c,1)
                mean_t_in_s = mean_t_in_s.view(c,1)
                var_s_in_t = var_s_in_t.view(c, 1)
                var_t_in_s = var_t_in_s.view(c, 1)
                cur_mean_source = cur_mean_source.view(c, 1)
                cur_mean_target = cur_mean_target.view(c, 1)
                cur_var_source = cur_var_source.view(c, 1)
                cur_var_target = cur_var_target.view(c, 1)

            mean_source = self.my_mean_w_s * cur_mean_source + (1-self.my_mean_w_s) * mean_s_in_t  # [c,1]
            mean_target = self.my_mean_w_t * cur_mean_target + (1-self.my_mean_w_t) * mean_t_in_s
            var_source = self.my_var_w_s * cur_var_source + (1-self.my_var_w_s) * var_s_in_t
            var_target = self.my_var_w_t * cur_var_target + (1-self.my_var_w_t) * var_t_in_s

            with torch.no_grad():
                self.running_mean_source = (1-self.momentum) * self.running_mean_source + self.momentum * mean_source.squeeze(1)
                self.running_mean_target = (1-self.momentum) * self.running_mean_target + self.momentum * mean_target.squeeze(1)
                self.running_var_source = (1-self.momentum) * self.running_var_source + self.momentum * var_source.squeeze(1)
                self.running_var_target = (1-self.momentum) * self.running_var_target + self.momentum * var_target.squeeze(1)

            z_source = (input_source - mean_source) / (var_source + self.eps).sqrt()
            z_target = (input_target - mean_target) / (var_target + self.eps).sqrt()
            if input.dim() == 4:
                gamma = self.weight.view(1,self.num_features,1,1)
                beta = self.bias.view(1,self.num_features,1,1)
                z_source, z_target = z_source.view(c, batch_size, h, w).permute(1,0,2,3).contiguous(), z_target.view(c, batch_size, h, w).permute(1,0,2,3).contiguous()
            else:
                gamma = self.weight.view(1, self.num_features)
                beta = self.bias.view(1, self.num_features)
                z_source, z_target = z_source.view(c, batch_size).permute(1,0).contiguous(), z_target.view(c, batch_size).permute(1,0).contiguous()
            z_source = gamma * z_source + beta
            z_target = gamma * z_target + beta

            z = torch.cat((z_source, z_target), dim=0)
            return z
        
        else:  ##test mode
            z = F.batch_norm(
                input, self.running_mean_target, self.running_var_target, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps)
            return z

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)
        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            lamda_key = prefix + 'lamda'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)
            # if lamda_key not in state_dict:
            # lamda = torch.Tensor(1).cuda()
            # lamda.data.fill_(0.1).long()
            # state_dict[lamda_key] = lamda
            # state_dict[lamda_key] = torch.tensor([0.1], dtype=torch.long)

        self._load_from_state_dict_from_pretrained_model(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class RN1d(_ReciprocalNorm):
    r"""Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size).

    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class RN2d(_ReciprocalNorm):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size).

    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class RN3d(_ReciprocalNorm):
    r"""Applies Batch Normalization over a 5D input (a mini-batch of 3D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size).

    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric Batch Normalization
    or Spatio-temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
