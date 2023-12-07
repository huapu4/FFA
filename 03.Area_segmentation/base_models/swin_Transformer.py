import tensorflow as tf
from base_models.keras_vision_transformer import swin_layers
from base_models.keras_vision_transformer import transformer_layers

from tensorflow.keras.layers import Dense, concatenate

layers = tf.keras.layers
backend = tf.keras.backend


class Swin_Transformer(object):
    def __init__(self, version='Swin_Transformer', dilation=None, **kwargs):
        """
        The implementation of Swin Transformer based on Tensorflow.
        :param version: 'Swin Transformer'
        :param kwargs:
            # filter_num_begin: number of channels in the first downsampling block; it is also the number of embedded dimensions
            # depth: the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level
            # stack_num_down: number of Swin Transformers per downsampling level
            # stack_num_up : number of Swin Transformers per upsampling level
            # patch_size:  Extract 2-by-2 patches from the input image. Height and width of the patch must be equal.
            # num_heads : number of attention heads per down/upsampling level
            # window_size: the size of attention window per down/upsampling level
            # num_mlp: number of MLP nodes within the Transformer
            # shift_window: Apply window shifting, i.e., Swin-MSA
        """
        super(Swin_Transformer, self).__init__(**kwargs)

        params = {'Swin_Transformer': {'filter_num_begin' : 128, 'depth' : 4, 'stack_num_down' : 2, 'stack_num_up' : 2,
                                       # 'patch_size' : (4, 4), 'num_heads' : [4, 8, 8, 16, 16, 32], 'window_size' : [8, 8, 8, 8, 8, 4],
                                       # 'patch_size': (4, 4), 'num_heads': [4, 8, 16, 32], 'window_size': [7, 7, 7, 7],
                                       'patch_size': (4, 4), 'num_heads': [4, 8, 16, 32], 'window_size': [8, 8, 8, 16],
                                       # 'patch_size': (4, 4), 'num_heads': [4, 8, 16, 16,32], 'window_size': [8, 8, 8, 8, 8],
                                       'num_mlp' : 512, 'shift_window' : True }
                  }
        self.version = version
        assert version in params
        self.params = params[version]

        if dilation is None:
            self.dilation = [1, 1]
        else:
            self.dilation = dilation
        assert len(self.dilation) == 2

    def __call__(self, inputs, output_stages='c5', **kwargs):
        """
        call for .
        :param inputs: a 4-D tensor.
        :param output_stages: str or a list of str containing the output stages.
        :param kwargs: other parameters.
        :return: the output of different stages.
        """
        dilation = self.dilation
        _, h, w, _ = backend.int_shape(inputs)


        '''
        The base of SwinUNET.
        '''
        # Compute number be patches to be embeded
        input_size = inputs.shape.as_list()[1:]
        num_patch_x = input_size[0] // self.params['patch_size'][0]
        num_patch_y = input_size[1] // self.params['patch_size'][1]

        # Number of Embedded dimensions
        embed_dim = self.params['filter_num_begin']

        depth_ = self.params['depth']

        X_skip = []

        X = inputs

        # Patch extraction
        X = transformer_layers.patch_extract(self.params['patch_size'])(X)

        # Embed patches to tokens
        X = transformer_layers.patch_embedding(num_patch_x * num_patch_y, embed_dim)(X)

        # The first Swin Transformer stack
        X = self.swin_transformer_stack(X, stack_num= self.params['stack_num_down'],
                                        embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y),
                                        num_heads= self.params['num_heads'][0],
                                        window_size=self.params['window_size'][0], num_mlp=self.params['num_mlp'],
                                        shift_window=self.params['shift_window'], name='swin_down0')
        X_skip.append(X)

        # Downsampling blocks
        for i in range(depth_ - 1):
            # Patch merging
            X = transformer_layers.patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim,
                                                 name='down{}'.format(i))(
                X)

            # update token shape info
            embed_dim = embed_dim * 2
            num_patch_x = num_patch_x // 2
            num_patch_y = num_patch_y // 2

            # Swin Transformer stacks
            # in stage3, Swin Transformer stacks has multi stacks
            if i == 1:
                stack_num_down_ = self.params['stack_num_down'] * 9
            else:
                stack_num_down_ = self.params['stack_num_up']
            X = self.swin_transformer_stack(X, stack_num=stack_num_down_,
                                            embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y),
                                            num_heads=self.params['num_heads'][i + 1],
                                            window_size=self.params['window_size'][i + 1],
                                            num_mlp=self.params['num_mlp'],
                                            shift_window=self.params['shift_window'],
                                            name='swin_down{}'.format(i + 1))

            # Store tensors for concat
            X_skip.append(X)

        # reverse indexing encoded tensors and hyperparams
        X_skip = X_skip[::-1]
        num_heads = self.params['num_heads'][::-1]
        window_size = self.params['window_size'][::-1]

        # upsampling begins at the deepest available tensor
        X = X_skip[0]

        # other tensors are preserved for concatenation
        X_decode = X_skip[1:]

        depth_decode = len(X_decode)

        for i in range(depth_decode):
            # Patch expanding
            X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y),
                                                   embed_dim=embed_dim, upsample_rate=2, return_vector=True)(X)

            # update token shape info
            embed_dim = embed_dim // 2
            num_patch_x = num_patch_x * 2
            num_patch_y = num_patch_y * 2

            # Concatenation and linear projection
            X = concatenate([X, X_decode[i]], axis=-1, name='swin_concat_{}'.format(i))
            X = Dense(embed_dim, use_bias=False, name='swin_concat_linear_proj_{}'.format(i))(X)

            # Swin Transformer stacks
            if i == 1:
                stack_num_up_ = self.params['stack_num_up'] * 9
            else:
                stack_num_up_ = self.params['stack_num_up']
            X = self.swin_transformer_stack(X, stack_num=stack_num_up_,
                                            embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y),
                                            num_heads=num_heads[i],
                                            window_size=window_size[i],
                                            num_mlp=self.params['num_mlp'],
                                            shift_window=self.params['shift_window'],
                                            name='swin_up{}'.format(i))

        # The last expanding layer; it produces full-size feature maps based on the patch size
        # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
        X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y),
                                               embed_dim=embed_dim, upsample_rate=self.params['patch_size'][0], return_vector=False)(X)

        return X



    def swin_transformer_stack(self, X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp, shift_window=True,
                               name=''):
        '''
        Stacked Swin Transformers that share the same token size.

        Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
        *Dropout is turned off.
        '''
        # Turn-off dropouts
        mlp_drop_rate = 0  # Droupout after each MLP layer
        attn_drop_rate = 0  # Dropout after Swin-Attention
        proj_drop_rate = 0  # Dropout at the end of each Swin-Attention block, i.e., after linear projections
        drop_path_rate = 0  # Drop-path within skip-connections

        qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
        qk_scale = None  # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor

        if shift_window:
            shift_size = window_size // 2
        else:
            shift_size = 0

        for i in range(stack_num):

            if i % 2 == 0:
                shift_size_temp = 0
            else:
                shift_size_temp = shift_size

            X = swin_layers.SwinTransformerBlock(dim=embed_dim, num_patch=num_patch, num_heads=num_heads,
                                                 window_size=window_size, shift_size=shift_size_temp, num_mlp=num_mlp,
                                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                 mlp_drop=mlp_drop_rate, attn_drop=attn_drop_rate, proj_drop=proj_drop_rate,
                                                 drop_path_prob=drop_path_rate,
                                                 name='name{}'.format(i))(X)
        return X