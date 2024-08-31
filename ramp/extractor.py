import torch
import torch.nn as nn

DIM=32
CHANNEL_DIM=5


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BasicEncoder4(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, multidim=False, channel_dim=CHANNEL_DIM):
        super(BasicEncoder4, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim
        self.channel_dim = channel_dim

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(channel_dim, DIM, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM, stride=1)
        self.in_planes_layer1 = self.in_planes
        self.layer2 = self._make_layer(2*DIM, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(2*DIM, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, output_dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, output_dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(output_dim, output_dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = output_dim
        return nn.Sequential(*layers)

    def _forward(self, x):
        b, n, c1, h1, w1 = x.shape
        x = x.view(b*n, c1, h1, w1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv2(x)

        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)
    

    def forward(self, x):
        return self._forward(x)


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell

    Reference from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py

    """

    def __init__(self, input_size, hidden_size, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, self.kernel_size, padding=self.padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.zeros(state_size, dtype=input_.dtype).to(input_.device),
                torch.zeros(state_size, dtype=input_.dtype).to(input_.device)
                )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class MergerLSTMsceneEncoder(nn.Module):
    def __init__(
            self, 
            evs_ch_dim=5, 
            img_ch_dim=3, 
            output_lstm_dim=15, 
            output_dim_f=128,
            output_dim_i=DIM, 
            norm_fn_fmap="instance",
            norm_fn_imap="none",
            kernel_size_superstate=1,
            ):
        super().__init__()

        # Common state for images and events
        self.evs_ch_dim = evs_ch_dim
        self.img_ch_dim = img_ch_dim
        self.kernel_size_superstate = kernel_size_superstate
        self.state_enc_input_size = 2*output_lstm_dim
        self.state_enc_out_size = output_lstm_dim
        self.hidden_size = output_lstm_dim
        padding = (kernel_size_superstate - 1)//2 
        self.padding = padding

        self.events_convlstm = nn.LSTM(input_size=evs_ch_dim, hidden_size=self.hidden_size, batch_first=True)
        self.image_convlstm = nn.LSTM(input_size=img_ch_dim,  hidden_size=self.hidden_size, batch_first=True)

        self.superstate_encoder = nn.Conv2d(
            in_channels=self.state_enc_input_size, 
            out_channels=self.state_enc_out_size, 
            kernel_size=kernel_size_superstate, 
            padding=padding)

        self.fmap_encoder = BasicEncoder4(output_dim=output_dim_f, norm_fn=norm_fn_fmap, channel_dim=self.state_enc_out_size)
        self.imap_encoder = BasicEncoder4(output_dim=output_dim_i, norm_fn=norm_fn_imap, channel_dim=self.state_enc_out_size)
        
        self.states_events, self.states_image, self.super_state = None, None, None

    def forward_superstate(self, data, prev_super_state=None):
        super_state = prev_super_state
        if super_state is None:
            super_state = torch.zeros_like(data)
        state_composition = torch.concat((super_state, data),dim=0)
        super_state = self.superstate_encoder(state_composition)
        return super_state

    def forward(self, events, images, reinit_hidden=False):
        if reinit_hidden:
            self.states_events, self.states_image, self.super_state = None, None, None

        B,T_events,C_events,H,W = events.shape
        B,T_images,C_images,H,W = images.shape
        events_as_seq = events.permute(0, 3, 4, 1, 2).contiguous().view(B*H*W, T_events, C_events)
        image_as_seq = images.permute(0, 3, 4, 1, 2).contiguous().view(B*H*W, T_images, C_images)

        out_events_flat, self.states_events = self.events_convlstm(input=events_as_seq, hx=self.states_events)
        out_images_flat, self.states_image = self.image_convlstm(input=image_as_seq, hx=self.states_image)

        # TODO reshaping of this is not necessary here
        out_events = out_events_flat.view(B,H,W,T_events,self.hidden_size).permute(0, 3, 4, 1, 2)
        out_images = out_images_flat.view(B,H,W,T_images,self.hidden_size).permute(0, 3, 4, 1, 2)

        if isinstance(self.superstate_encoder, nn.Conv2d):
            super_states = []
            for ind, (ev_embed, img_embed) in enumerate(zip(out_events[0],out_images[0])):
                # TODO put these variables outside using a mask
                events_are_present = torch.any(events[:,ind] != 0)
                image_is_present = torch.any(images[:,ind] != 0)
                if events_are_present:
                    self.super_state = self.forward_superstate(ev_embed, prev_super_state=self.super_state)
                if image_is_present:
                    self.super_state = self.forward_superstate(img_embed, prev_super_state=self.super_state)
                super_states.append(self.super_state)
        else:
            raise NotImplementedError
        
        super_states = torch.stack(super_states,dim=0)[None,...]
        lstms_states = [(out_events, out_images)]

        fmap = self.fmap_encoder(super_states)
        imap = self.imap_encoder(super_states)

        return fmap, imap, lstms_states


######## Basic blocks section ########

class MultiScaleBasicEncoder4(BasicEncoder4):
    def __init__(self, output_dim=128,  internal_input_dimensions=None, **kwargs):
        super(BasicEncoder4, self).__init__()
        super(MultiScaleBasicEncoder4, self).__init__(**kwargs)
        if internal_input_dimensions is None:
            internal_input_dimensions = [self.channel_dim]*3

        self.internal_input_dimensions = internal_input_dimensions
        self.in_planes = self.in_planes_layer1 + self.internal_input_dimensions[1]
        self.in_planes_layer2 = self.in_planes #inp dim layer 3 
        self.in_planes_layer3 = self.in_planes_layer2 + self.internal_input_dimensions[2]
        self.layer3 = self._make_layer(output_dim=2*DIM, stride=2)
        self.conv3 = nn.Conv2d(2*DIM + self.internal_input_dimensions[2], output_dim, kernel_size=1)

    def forward(self, x, x_down2, x_down4):
        b, n, c1, h1, w1 = x.shape
        x = x.view(b*n, c1, h1, w1)

        _, _, c2, h2, w2 = x_down2.shape
        x_down2 = x_down2.view(b*n, c2, h2, w2)

        _, _, c4, h4, w4 = x_down4.shape
        x_down4 = x_down4.view(b*n, c4, h4, w4)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x) # Scale 0.5
        x = torch.cat((x, x_down2), dim=1)

        x = self.layer3(x) # Scale 0.25
        x = torch.cat((x, x_down4), dim=1)

        x = self.conv3(x)

        _, c3, h3, w3 = x.shape
        return x.view(b, n, c3, h3, w3)


class LSTMEncoder(nn.Module):
    def __init__(
        self, 
        in_channels, 
        downsample_scale=0, 
        out_channels=15, 
        batch_norm_momentum=0.1,
        activation_fn=None,
        normalization_type=None,
        ):
        super().__init__()

        self.kernel_size_1 = downsample_scale+1
        self.hidden_size_lstm = out_channels
        self.activation_fn = activation_fn
        self.stride = downsample_scale
        self.norm = normalization_type
        self.bn_mom = batch_norm_momentum
        self.padding = 1
        
        if downsample_scale <= 1:
            self.kernel_size_1 = 1
            self.stride = 1
            self.padding = 0

        self.conv_1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=self.kernel_size_1, 
            stride=self.stride, 
            padding=self.padding
            )

        if self.activation_fn is not None:
            activation = getattr(torch, activation)
            self.conv_1 = nn.Sequential(self.conv_1, activation)

        self.convlstm = nn.LSTM(
            input_size=in_channels, 
            hidden_size=self.hidden_size_lstm, 
            batch_first=True
            )

        if self.norm is not None and self.norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=self.bn_mom)
        elif self.norm is not None and self.norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)
        elif self.norm is None:
            self.norm_layer = nn.Sequential()
        else:
            raise NotImplementedError("Not supported normalization type")
    
    @staticmethod
    def to_sequence(tensor):
        B,T,C,H,W = tensor.shape
        return tensor.permute(0, 3, 4, 1, 2).contiguous().view(B*H*W, T, C), tensor.shape
    
    @staticmethod
    def from_sequence_to_original(orig_shape, sequence, hidden_size):
        B,T,C,H,W = orig_shape
        return sequence.view(B,H,W,T,hidden_size).permute(0, 3, 4, 1, 2)
    
    def forward_lstm(self, x):
        s, shape = self.to_sequence(x)
        s, state = self.convlstm(s)
        x = self.from_sequence_to_original(sequence=s, orig_shape=shape, hidden_size=self.hidden_size_lstm)
        x = self.norm_layer(x)
        return x, state

    def forward(self, x):
        x = self.conv_1(x.squeeze(0)).unsqueeze(0)
        return self.forward_lstm(x)
    
    def hierarchical_forward(self, prev_feature):
        x = self.conv_1(prev_feature.squeeze(0)).unsqueeze(0)
        out, state = self.forward_lstm(x)
        return x, out, state


class SuperStateEncoder(nn.Module):
    # TODO expand the way we compute super state
    def __init__(self, kernel_size, out_channels=15, norm_superstate=False):
        super().__init__()

        self.input_size = 2*out_channels
        self.out_size = out_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1)//2 
        self.encoder = nn.Conv2d(in_channels=self.input_size, out_channels=self.out_size, kernel_size=self.kernel_size, padding=self.padding)
        self.instance_norm_layer = nn.InstanceNorm2d(num_features=self.out_size)  # num_features = number of channels
        self.norm_superstate = norm_superstate 
    
    def forward(self, data, prev_super_state=None):
        super_state = prev_super_state
        if super_state is None:
            super_state = torch.zeros_like(data)
        state_composition = torch.concat((super_state, data),dim=0)
        super_state = self.encoder(state_composition)
        return super_state
        
    @staticmethod
    def forward_event_images_to_superstate(out_events, out_images, event_mask, image_mask, ev_encoder, im_encoder, super_state):
        out_events = out_events.squeeze(0)
        out_images = out_images.squeeze(0)
        if super_state is not None:
            not_same_shape = super_state.shape not in (out_events[0].shape, out_images[0].shape)
            super_state = super_state.squeeze() if not_same_shape else super_state
            
        super_states = []
        for ind, (ev_embed, img_embed) in enumerate(zip(out_events,out_images)):
            if event_mask[ind]:
                super_state = ev_encoder.forward(ev_embed, prev_super_state=super_state)
            if image_mask[ind]:
                super_state = im_encoder.forward(img_embed, prev_super_state=super_state)
            super_states.append(super_state)
        return torch.stack(super_states,dim=0)[None,...]
        
    @staticmethod
    def forward_superstate(
        out_events, out_images, mask, ev_encoder, im_encoder, prev_super_state
        ):
        out_events = out_events.squeeze(0)
        out_images = out_images.squeeze(0)
        mask = mask.squeeze(0)

        if prev_super_state is not None:
            prev_super_state = prev_super_state.squeeze()
        
        data_len = out_events.shape[0]
        
        super_states = []
        ind_im = 0
        for ind_ev in range(data_len):
            prev_super_state = ev_encoder.forward(data=out_events[ind_ev], prev_super_state=prev_super_state)
            # TODO hacky way to check if mask is a tensor or not
            supervise = mask[ind_ev].item() if mask.dim() > 0 else mask.item()
            # supervise = mask[ind_ev] if len([mask]) > 1 else mask.item()
            if supervise:
                prev_super_state = im_encoder.forward(data=out_images[ind_im], prev_super_state=prev_super_state)
                ind_im += 1
                super_states.append(prev_super_state)

        # normalize super state stack with batch normalization
        all_super_states = prev_super_state[None,...] if not super_states else torch.stack(super_states,dim=0)
        if ev_encoder.norm_superstate or im_encoder.norm_superstate:
            norm_super_states = ev_encoder.instance_norm_layer(all_super_states)
        else:
            norm_super_states = all_super_states

        return norm_super_states
    

######## Multi Scale merger section ########

class MultiScaleMergerDoubleNet(nn.Module):
    def __init__(
            self, 
            evs_ch_dim, 
            img_ch_dim, 
            lstm_dim=16, 
            output_dim_f=128, 
            output_dim_i=DIM, 
            norm_fn_fmap="instance",
            norm_fn_imap="none",
            kernel_size_superstate=1,
            activation_fn=None,
            normalization_type=None,
            norm_superstate=False,
            ):
        super().__init__()

        # Common state for images and events
        scales = [1, 2, 4]
        self.scales = scales
        self.evs_ch_dim = evs_ch_dim
        self.img_ch_dim = img_ch_dim
        self.hidden_size = lstm_dim

        assert len(scales) == 3

        self.states_events, self.states_images, self.super_states = [], [], []
        self.ev_encoders = nn.ModuleList()
        self.im_encoders = nn.ModuleList()
        self.super_state_ev_encoder = nn.ModuleList()
        self.super_state_im_encoders = nn.ModuleList()
        self.internal_dimensions = []
        for scale in scales:
            internal_lstm_dim = lstm_dim * scale
            self.internal_dimensions.append(internal_lstm_dim)
            self.ev_encoders.append(LSTMEncoder(
                in_channels=evs_ch_dim, 
                downsample_scale=scale, 
                out_channels=internal_lstm_dim, 
                activation_fn=activation_fn, 
                normalization_type=normalization_type
                ))
            self.im_encoders.append(LSTMEncoder(
                in_channels=img_ch_dim, 
                downsample_scale=scale, 
                out_channels=internal_lstm_dim, 
                activation_fn=activation_fn, 
                normalization_type=normalization_type
                ))

            self.super_state_ev_encoder.append(SuperStateEncoder(
                kernel_size=kernel_size_superstate, out_channels=internal_lstm_dim, norm_superstate=norm_superstate)
                )
            self.super_state_im_encoders.append(SuperStateEncoder(
                kernel_size=kernel_size_superstate, out_channels=internal_lstm_dim, norm_superstate=norm_superstate)
                )

            self.super_states.append(None)

        self.fmap_encoder = MultiScaleBasicEncoder4(
            output_dim=output_dim_f, 
            norm_fn=norm_fn_fmap, 
            channel_dim=lstm_dim,
            internal_input_dimensions=self.internal_dimensions
            )
        self.imap_encoder = MultiScaleBasicEncoder4(
            output_dim=output_dim_i, 
            norm_fn=norm_fn_imap, 
            channel_dim=lstm_dim,
            internal_input_dimensions=self.internal_dimensions
            )
    
    def forward(self, events, images, mask, reinit_hidden=False):
        for scale, (encoders) in enumerate(zip(self.ev_encoders, self.im_encoders, self.super_state_ev_encoder, self.super_state_im_encoders)):
            if reinit_hidden:
                self.super_states[scale] = None

            ev_encoder, img_encoder, ss_ev_encoder, ss_im_encoder = encoders
            out_events, out_images = None, None

            # TODO if not real-time, encode before all tensors all at once
            out_events, state_events = ev_encoder.forward(x=events)
            out_images, state_images = img_encoder.forward(x=images)

            super_state = SuperStateEncoder.forward_superstate(               
                out_events=out_events, 
                out_images=out_images, 
                mask=mask,
                prev_super_state=self.super_states[scale], 
                ev_encoder=ss_ev_encoder,
                im_encoder=ss_im_encoder,
                )

            self.super_states[scale]=super_state[None,...]

        fmap = self.fmap_encoder(x=self.super_states[0], x_down2=self.super_states[1], x_down4=self.super_states[2])
        imap = self.imap_encoder(x=self.super_states[0], x_down2=self.super_states[1], x_down4=self.super_states[2])

        return fmap, imap
