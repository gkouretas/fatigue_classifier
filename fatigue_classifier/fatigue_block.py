import keras

@keras.saving.register_keras_serializable()
class FatigueLSTMBlock(keras.models.Model):
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def __init__(self, weight: float | None, mask: float | None = None, lstm_kwargs: dict = {}, conv_layers_kwargs: list[dict] | None = None, **kwargs):
        super().__init__(**kwargs)

        self._weight = weight
        self._mask = mask
        self._lstm_kwargs = lstm_kwargs
        self._conv_layers_kwargs = conv_layers_kwargs
        
        if conv_layers_kwargs is not None:
            self._conv_layers: list[keras.layers.Conv1D] = []
            for conv_kwargs in conv_layers_kwargs:
                self._conv_layers.append(keras.layers.Conv1D(**conv_kwargs))
        else:
            self._conv_layers = None
        
        if mask is None:
            self._mask_layer = None
        else:
            self._mask_layer = keras.layers.Masking(mask_value=mask)

        self._lstm_layer = keras.layers.LSTM(
            **lstm_kwargs
        )

        self._classification_layer = keras.layers.Dense(
            1,
            activation="sigmoid",
        )

        self._scale_layer = None
        if weight is not None:
            # Hard-coded scale layer
            self._scale_layer = keras.layers.Lambda(
                lambda x: x * self._weight
            )

        self._output_shape = None
        self._last_pred = None
    
    @property
    def weight(self):
        return self._weight
    
    @property
    def mask(self):
        return self._mask

    @property
    def last_prediction(self):
        return self._last_pred

    @property
    def output_layer(self):
        return self._classification_layer
    
    @property
    def output_shape(self):
        return self._output_shape
    
    def get_config(self):
        return {
            "weight": self._weight,
            "mask": self._mask,
            "lstm_kwargs": self._lstm_kwargs,
            "conv_layers_kwargs": self._conv_layers_kwargs
        }

    def build(self, input_shape):
        if self._conv_layers is not None:
            for conv_layer in self._conv_layers:
                conv_layer.build(input_shape)
                input_shape = conv_layer.compute_output_shape(input_shape)
        
        if self._mask_layer is not None:
            self._mask_layer.build(input_shape)
            input_shape = self._mask_layer.compute_output_shape(input_shape)

        self._lstm_layer.build(input_shape)
        input_shape = self._lstm_layer.compute_output_shape(input_shape)

        self._classification_layer.build(input_shape)
        input_shape = self._classification_layer.compute_output_shape(input_shape)

        if self._scale_layer is not None:
            self._scale_layer.build(input_shape)
            input_shape = self._scale_layer.compute_output_shape(input_shape)

        self._output_shape = input_shape

    def call(self, inputs):
        x = inputs
        if self._conv_layers is not None:
            for conv_layer in self._conv_layers:
                x = conv_layer(x)
                
        if self._mask_layer is not None:
            x = self._mask_layer(x)

        x = self._lstm_layer(x)
        x = self._classification_layer(x)
        if self._scale_layer is not None:
            x = self._scale_layer(x)

        return x
    
    def compute_output_shape(self, input_shape):
        if self._output_shape is None:
            self.build(input_shape)
            
        return self._output_shape
