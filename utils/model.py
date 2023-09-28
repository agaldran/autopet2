import sys
import torch
from monai.networks.nets import UNet, UNETR, SwinUNETR, DynUNet
from monai.networks.blocks import UnetrUpBlock, UnetOutBlock




def num_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def get_kernels_strides(patch_size, spacings):
    input_size = patch_size
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, patch_size)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(patch_size, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        patch_size = [i / j for i, j in zip(patch_size, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides

def get_network(patch_size, spacings, in_channels, n_classes, n_heads=1, deep_supr_num=0):
    kernels, strides = get_kernels_strides(patch_size, spacings)

    deep_supervision = False if deep_supr_num == 0 else True
    assert n_heads >= 1
    if n_heads == 1:
        net = DynUNet(spatial_dims=3, in_channels=in_channels, out_channels=n_classes, kernel_size=kernels, strides=strides,
            upsample_kernel_size=strides[1:], norm_name="instance", deep_supervision=deep_supervision, deep_supr_num=deep_supr_num)
    else:
        net = MultiHeadDynUnet(n_heads=n_heads, spatial_dims=3, in_channels=in_channels, out_channels=n_classes,
                               kernel_size=kernels, strides=strides, upsample_kernel_size=strides[1:])

    return net

class MultiHeadDynUnet(DynUNet):
    def __init__(self, n_heads, spatial_dims, in_channels, out_channels, kernel_size, strides, upsample_kernel_size):
        super(MultiHeadDynUnet, self).__init__(spatial_dims=spatial_dims, in_channels=in_channels,
                                               out_channels=out_channels, kernel_size=kernel_size,
                                               strides=strides, upsample_kernel_size=upsample_kernel_size,
                                               norm_name="instance", deep_supervision=False) # do not allow DS here, it'd be a mess
        self.n_heads = n_heads
        self.pred_heads = torch.nn.ModuleList([self.get_output_block(0) for _ in range(self.n_heads)])

    def forward(self, x):
        out = self.skip_layers(x)
        head_predictions = [self.pred_heads[i](out) for i in range(self.n_heads)]
        out = torch.stack(head_predictions, dim=0)  # stacked along the first dimension
        if self.training: # return n_heads x logits
            return out  # BS x n_heads x X x Y x Z
        else:
            return out.sigmoid().mean(dim=0)  # BS x X x Y x Z mean over heads AFTER sigmoiding them

class MultiHead_SwinUNETR(SwinUNETR):
    def __init__(self, img_size, in_channels, n_classes, feature_size, n_heads=2, lite=True, pretrained=True):
        self.n_classes = n_classes
        self.feature_size = feature_size
        self.n_heads = n_heads
        self.lite = lite
        # temporarily build the model with out_channels=14 for pretrained weight loading
        if pretrained:
            SwinUNETR.__init__(self, img_size=img_size, in_channels=in_channels,
                                 out_channels=14, feature_size=feature_size)
            # load pretrained weights
            if feature_size == 12:
                state_dict = torch.load('pretrained_weights/swin_unetr_tiny_btcv.pt')["state_dict"]
            elif feature_size == 24:
                state_dict = torch.load('pretrained_weights/swin_unetr_small_btcv.pt')["state_dict"]
            elif feature_size == 48:
                state_dict = torch.load('pretrained_weights/swin_unetr_base_btcv.pt')["state_dict"]
            else: sys.exit('not a valid feature size for pretrained weights')

            # grab pretrained weights of input layer(s), which have one channel
            input_weight_name1 = 'swinViT.patch_embed.proj.weight'
            input_weight_name2 = 'encoder1.layer.conv1.conv.weight'
            input_weight_name3 = 'encoder1.layer.conv3.conv.weight'

            input_weight1 = state_dict[input_weight_name1]
            input_weight2 = state_dict[input_weight_name2]
            input_weight3 = state_dict[input_weight_name3]

            # stack as many times as wanted input channels, manipulate the pretrained state_dict
            state_dict[input_weight_name1] = torch.cat(in_channels*[input_weight1], dim=1)
            state_dict[input_weight_name2] = torch.cat(in_channels*[input_weight2], dim=1)
            state_dict[input_weight_name3] = torch.cat(in_channels*[input_weight3], dim=1)
            # now we can load our model
            self.load_state_dict(state_dict)
            # no need to redefine outblock since we are removing it below
            # model.out = UnetOutBlock(spatial_dims=3, in_channels=feat_sz, out_channels=n_classes)
        else:
            SwinUNETR.__init__(self, img_size=img_size, in_channels=in_channels,
                                 out_channels=n_classes, feature_size=feature_size)

        self.norm_name = "instance"
        # no need for these right?
        del self.decoder1
        del self.out

        self.heads = torch.nn.ModuleList([self.build_head() for _ in range(self.n_heads)])

        if self.lite:
            self.last_dec = self.build_neck()
        else:
            self.necks = torch.nn.ModuleList([self.build_neck() for _ in range(self.n_heads)])

    def build_head(self):
        return UnetOutBlock(spatial_dims=3, in_channels=self.feature_size, out_channels=self.n_classes)
    def build_neck(self):
        return UnetrUpBlock(spatial_dims=3, in_channels=self.feature_size, out_channels=self.feature_size,
                            kernel_size=3, upsample_kernel_size=2, norm_name=self.norm_name, res_block=True)

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)

        if self.lite:
            out1 = self.last_dec(dec0, enc0)
            head_predictions = [self.heads[i](out1) for i in range(self.n_heads)]
        else:
            head_predictions = [self.heads[i](self.necks[i](dec0, enc0)) for i in range(self.n_heads)]

        out = torch.stack(head_predictions, dim=0)  # stacked along the first dimension

        if self.training: # return n_heads x logits
            return out  # n_heads x batch_size x num_classes
        else:
            return out.sigmoid().mean(dim=0)  # mean over heads AFTER sigmoiding them

def get_model(model_name, patch_size=(0,0,0), pretrained=False, n_classes=1, in_channels=2, n_heads=1):
    patch_size = tuple(patch_size)

    if model_name == 'micro_unet':

        model = UNet(spatial_dims=3, in_channels=in_channels, out_channels=n_classes,
                     channels=(32, 64), strides=(2,), num_res_units=0)
        if pretrained:
            print('No pretrained weights for {}'.format(model_name))

    elif model_name == 'dynunet':
        if patch_size == (0, 0, 0): patch_size = (128, 128, 96)
        spacings = 2.04, 2.04, 3.00
        deep_sup = 0
        model = get_network(patch_size=patch_size, spacings=spacings, in_channels=in_channels,
                            n_classes=n_classes, deep_supr_num=deep_sup, n_heads=n_heads)
        if pretrained:
            print('No pretrained weights for {}'.format(model_name))

    elif model_name == 'swinunetr_tiny':
        if patch_size == (0, 0, 0): patch_size = (96, 96, 96)
        feat_sz = 12
        if n_heads == 1:
            if pretrained: # weights from training on btcv, its ct but meh
                model = SwinUNETR(img_size=patch_size, in_channels=in_channels, out_channels=14, feature_size=feat_sz, use_checkpoint=False)
                state_dict = torch.load('pretrained_weights/swin_unetr_tiny_btcv.pt')["state_dict"]
                # grab pretrained weights of input layer(s), which have one channel
                input_weight_name1 = 'swinViT.patch_embed.proj.weight'
                input_weight_name2 = 'encoder1.layer.conv1.conv.weight'
                input_weight_name3 = 'encoder1.layer.conv3.conv.weight'

                input_weight1 = state_dict[input_weight_name1]
                input_weight2 = state_dict[input_weight_name2]
                input_weight3 = state_dict[input_weight_name3]

                # stack as many times as wanted input channels, manipulate the pretrained state_dict
                state_dict[input_weight_name1] = torch.cat(in_channels * [input_weight1], dim=1)
                state_dict[input_weight_name2] = torch.cat(in_channels * [input_weight2], dim=1)
                state_dict[input_weight_name3] = torch.cat(in_channels * [input_weight3], dim=1)
                # now we can load our model
                model.load_state_dict(state_dict)
                print('successfully loaded pretrained weights for swinunetr_tiny')
                model.out = UnetOutBlock(spatial_dims=3, in_channels=feat_sz, out_channels=n_classes)
            else:
                model = SwinUNETR(img_size=patch_size, in_channels=in_channels, out_channels=n_classes, feature_size=feat_sz, use_checkpoint=False)

        else:
            model = MultiHead_SwinUNETR(img_size=patch_size, in_channels=in_channels, n_classes=1, n_heads=n_heads, feature_size=feat_sz, lite=True, pretrained=pretrained)

    elif model_name == 'swinunetr_small':
        if patch_size == (0, 0, 0): patch_size = (96, 96, 96)
        feat_sz = 24
        if n_heads == 1:
            if pretrained: # weights from training on btcv, its ct but meh
                model = SwinUNETR(img_size=patch_size, in_channels=in_channels, out_channels=14, feature_size=feat_sz, use_checkpoint=False)
                state_dict = torch.load('pretrained_weights/swin_unetr_small_btcv.pt')["state_dict"]
                # grab pretrained weights of input layer(s), which have one channel
                input_weight_name1 = 'swinViT.patch_embed.proj.weight'
                input_weight_name2 = 'encoder1.layer.conv1.conv.weight'
                input_weight_name3 = 'encoder1.layer.conv3.conv.weight'

                input_weight1 = state_dict[input_weight_name1]
                input_weight2 = state_dict[input_weight_name2]
                input_weight3 = state_dict[input_weight_name3]

                # stack as many times as wanted input channels, manipulate the pretrained state_dict
                state_dict[input_weight_name1] = torch.cat(in_channels * [input_weight1], dim=1)
                state_dict[input_weight_name2] = torch.cat(in_channels * [input_weight2], dim=1)
                state_dict[input_weight_name3] = torch.cat(in_channels * [input_weight3], dim=1)
                # now we can load our model
                model.load_state_dict(state_dict)
                print('successfully loaded pretrained weights for swinunetr_small')
                model.out = UnetOutBlock(spatial_dims=3, in_channels=feat_sz, out_channels=n_classes)
            else:
                model = SwinUNETR(img_size=patch_size, in_channels=in_channels, out_channels=n_classes, feature_size=feat_sz, use_checkpoint=False)
        else:
            model = MultiHead_SwinUNETR(img_size=patch_size, in_channels=in_channels, n_classes=1, n_heads=n_heads, feature_size=feat_sz, lite=True, pretrained=pretrained)

    elif model_name == 'swinunetr_base':
        if patch_size == (0, 0, 0): patch_size = (96, 96, 96)
        feat_sz = 48
        if n_heads == 1:
            if pretrained: # weights from training on btcv, its ct but meh
                model = SwinUNETR(img_size=patch_size, in_channels=in_channels, out_channels=14, feature_size=feat_sz, use_checkpoint=False)
                state_dict = torch.load('pretrained_weights/swin_unetr_base_btcv.pt')["state_dict"]
                # grab pretrained weights of input layer(s), which have one channel
                input_weight_name1 = 'swinViT.patch_embed.proj.weight'
                input_weight_name2 = 'encoder1.layer.conv1.conv.weight'
                input_weight_name3 = 'encoder1.layer.conv3.conv.weight'

                input_weight1 = state_dict[input_weight_name1]
                input_weight2 = state_dict[input_weight_name2]
                input_weight3 = state_dict[input_weight_name3]

                # stack as many times as wanted input channels, manipulate the pretrained state_dict
                state_dict[input_weight_name1] = torch.cat(in_channels * [input_weight1], dim=1)
                state_dict[input_weight_name2] = torch.cat(in_channels * [input_weight2], dim=1)
                state_dict[input_weight_name3] = torch.cat(in_channels * [input_weight3], dim=1)
                # now we can load our model
                model.load_state_dict(state_dict)
                print('successfully loaded pretrained weights for swinunetr_base')
                model.out = UnetOutBlock(spatial_dims=3, in_channels=feat_sz, out_channels=n_classes)
            else:
                model = SwinUNETR(img_size=patch_size, in_channels=in_channels, out_channels=n_classes, feature_size=feat_sz, use_checkpoint=False)
        else:
            model = MultiHead_SwinUNETR(img_size=patch_size, in_channels=in_channels, n_classes=1, n_heads=n_heads, feature_size=48, lite=True, pretrained=pretrained)

    else: sys.exit('Model not implemented')

    print('#parameters = {}'.format(num_parameters(model)))
    setattr(model, 'model_name', model_name)
    setattr(model, 'n_classes', n_classes)
    setattr(model, 'patch_size', patch_size)
    setattr(model, 'n_heads', n_heads)

    return model

if __name__ == "__main__":
    model_name, n_heads = 'dynunet', 1
    model = get_model(model_name, patch_size=(128,128,64), pretrained=False, n_classes=1, in_channels=2, n_heads=n_heads)
    sys.exit()


    model_name = 'micro_unet'
    model = get_model(model_name, pretrained=False, n_classes=1, in_channels=2, n_heads=1)
    print(model_name, model(torch.randn((4,2,96,96,96))).shape)
    model_name, n_heads = 'dynunet', 1
    model = get_model(model_name, patch_size=(0, 0, 0), pretrained=False, n_classes=1, in_channels=2, n_heads=n_heads)
    print(model_name, n_heads, model(torch.randn((4,2,96,96,96))).shape)

    model_name, n_heads = 'dynunet', 4
    model = get_model(model_name, patch_size=(0, 0, 0), pretrained=False, n_classes=1, in_channels=2, n_heads=n_heads)
    print(model_name, n_heads, model(torch.randn((4,2,96,96,96))).shape)

    model_name, n_heads = 'swinunetr_tiny', 1
    model = get_model(model_name, patch_size=(0, 0, 0), pretrained=False, n_classes=1, in_channels=2, n_heads=n_heads)
    print(model_name, n_heads, model(torch.randn((4,2,96,96,96))).shape)

    model_name, n_heads = 'swinunetr_tiny', 4
    model = get_model(model_name, patch_size=(0, 0, 0), pretrained=False, n_classes=1, in_channels=2, n_heads=n_heads)
    print(model_name, n_heads, model(torch.randn((4,2,96,96,96))).shape)

    model_name, n_heads = 'swinunetr_small', 1
    model = get_model(model_name, patch_size=(0, 0, 0), pretrained=False, n_classes=1, in_channels=2, n_heads=n_heads)
    print(model_name, n_heads, model(torch.randn((4,2,96,96,96))).shape)

    model_name, n_heads = 'swinunetr_small', 4
    model = get_model(model_name, patch_size=(0, 0, 0), pretrained=False, n_classes=1, in_channels=2, n_heads=n_heads)
    print(model_name, n_heads, model(torch.randn((4,2,96,96,96))).shape)
