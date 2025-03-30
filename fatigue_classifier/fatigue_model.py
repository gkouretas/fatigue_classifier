import tensorflow as tf
import keras
import numpy as np
import os

from pathlib import Path

from numpy.typing import NDArray
from fatigue_block import FatigueLSTMBlock
from typing import TypedDict

class CompiledInputShape(TypedDict):
    fs: float
    max_duration_sec: float # TODO(george): remove?

    size: int
    window: int
    stride: int
    shape: tuple[int, int]
    
class InputArgs(TypedDict):
    fs: float
    window_size_sec: float
    stride_sec: float
    max_duration_sec: float

@keras.saving.register_keras_serializable()
class FatigueClassifier(keras.models.Model):
    @staticmethod
    def input_signal_shape(args: InputArgs) -> CompiledInputShape:
        print(args)
        _size = int(args.get("fs")*args.get("max_duration_sec"))
        _window = int(args.get("fs")*args.get("window_size_sec"))
        _stride = int(args.get("fs")*args.get("stride_sec"))

        return CompiledInputShape(
            fs=args.get("fs"),
            max_duration_sec=args.get("max_duration_sec"),
            size=_size,
            window=_window,
            stride=_stride,
            shape=(int((_size-_window)/_stride), _window)
        )
    
    @staticmethod
    def preprocess_signal(signal: NDArray, args: InputArgs, pad_value: float = 0.0):
        # TODO(george): this fails if stride is not divisible by the window size. 
        # Add a check for this
        _input_shape = FatigueClassifier.input_signal_shape(args)
        _output_signal = np.zeros((signal.shape[0], _input_shape.get("shape")[0], _input_shape.get("shape")[1]))
        
        for i in range(signal.shape[0]):
            for j, k in enumerate(range(0, _input_shape.get("size")-_input_shape.get("window"), _input_shape.get("stride"))):
                if k >= signal[i].size:
                    _output_signal[i][j] = np.ones(shape = (1, _input_shape.get("window"))) * pad_value
                elif k+_input_shape.get("window") >= signal[i].size:
                    _output_signal[i][j] = np.concatenate(
                        [np.array(signal[i][k:signal[i].size]).reshape((1, signal[i][k:signal[i].size].size)), np.ones(shape = (1, _input_shape.get("window")-(signal[i].size-k))) * pad_value],
                        axis = 1
                    )
                else:
                    _output_signal[i][j] = signal[i][k:k+_input_shape.get("window")]

        return _output_signal, _input_shape
    
    @staticmethod
    def preprocess_labels(labels: NDArray, fs: float, window_size: float, stride: float, max_duration_sec: float, pad_value: float = 0.0):
        # TODO(george): this fails if stride is not divisible by the window size. Add a check for this
        _size = int(fs*max_duration_sec)
        _window = int(fs*window_size)
        _stride = int(fs*stride)

        _output_signal = np.zeros((labels.shape[0], int((_size-_window)/_stride)))
        for i in range(labels.shape[0]):
            for j, k in enumerate(range(0, _size-_window, _stride)):
                if k+_window >= labels[i].size:
                    _output_signal[i][j] = pad_value
                else:
                    _output_signal[i][j] = labels[i][k+_window]

        return _output_signal

    @staticmethod
    @keras.utils.register_keras_serializable()
    def loss(y_true, y_pred, mask=0.0):
        if mask is not None:
            # Get boolean tensor where true = valid, false = invalid
            y_true_mask = tf.cast(y_true != mask, dtype=tf.float32)

            n_elem = tf.reduce_sum(y_true_mask)

            return tf.cond(tf.equal(n_elem, 0.0), lambda: 0.0, lambda: tf.reduce_sum(tf.square(y_true - y_pred) * y_true_mask) / n_elem)

        else:
            return keras.losses.mean_squared_error(y_true, y_pred)
        
    @classmethod
    def from_file(cls, path: str | Path) -> "FatigueClassifier":
        if os.path.exists(path):
            model = keras.models.load_model(
                path, 
                custom_objects={
                    "FatigueLSTMBlock": FatigueLSTMBlock,
                    "loss": cls.loss
                }
            )

            assert isinstance(model, cls), \
                f"Model is not an instance of {cls}"
            
            return model
        else:
            raise ValueError(f"Path {path} does not exist")
        
    @classmethod
    def from_config(cls, config: dict):
        # Explicitly deserialize lstm blocks
        config["lstm_blocks"] = [
            FatigueLSTMBlock.from_config(x) for x in config["lstm_blocks"]
        ]

        return cls(**config)

    def __init__(self, lstm_blocks: list[FatigueLSTMBlock], input_shapes: list[CompiledInputShape] | None = None, **kwargs):
        self._kwargs = kwargs
        super().__init__(**kwargs)
        
        self._lstm_blocks = lstm_blocks
        self._input_shapes = input_shapes
        
        assert all(x.weight is None for x in self._lstm_blocks) or np.sum([x.weight for x in lstm_blocks]) == 1.0, \
            "Sum of the weights must be equal to 1.0, or weights must all be None"
        assert np.all(np.asarray([x.mask for x in lstm_blocks]) == lstm_blocks[0].mask), \
            "All LSTM blocks must use the same mask"

        self._mask = lstm_blocks[0].mask
        self._hard_coded_weights = self._lstm_blocks[0].weight is not None

        self._reshape_layer: keras.layers.Concatenate | None
        self._classification_layer: keras.layers.Dense | keras.layers.Add | None

    def get_input_shape(self, name_substring: str) -> tuple[int, CompiledInputShape] | tuple[None, None]:
        if self._input_shapes is None: return None

        for i, block in enumerate(self._lstm_blocks):
            if name_substring in block.name: return i, self._input_shapes[i]

        return None, None

    def get_config(self):
        config = {
            "lstm_blocks": [x.get_config() for x in self._lstm_blocks],
            "input_shapes": self._input_shapes
        }

        config.update(self._kwargs)
        return config

    def build(self, input_shape):                
        output_shapes = []
        for lstm_block, input_vector in zip(self._lstm_blocks, input_shape):
            output_shapes.append(lstm_block.compute_output_shape(input_vector))

        if self._input_shapes is not None:
            assert len(self._lstm_blocks) == len(self._input_shapes), \
                f"Mismatch in lengths of LSTM blocks and input shapes ({len(self._lstm_blocks)} != {len(self._input_shapes)})"

            for block, input_shape_actual, input_shape_expected in zip(self._lstm_blocks, input_shape, self._input_shapes):
                # Assert shape is equivalent. Ignore first element (batch size)
                assert block.input_shape[1:] == input_shape_actual[1:] == input_shape_expected.get("shape"), \
                    f"Mismatch in input shape ({block.input_shape[1:]} != {input_shape_actual} != {input_shape_expected.get('shape')})"


        if not self._hard_coded_weights:
            # Concatenate LSTM blocks
            self._reshape_layer = keras.layers.Concatenate(axis=-1)
            self._reshape_layer.build(output_shapes)
            input_shape = self._reshape_layer.compute_output_shape(output_shapes)

            # Single classification of fatigue from 0-1
            self._classification_layer = keras.layers.Dense(1, activation="sigmoid", name="fatigue_classification_layer")
            self._classification_layer.build(input_shape)
        else:
            # Classification layer is to simply sum the blocks together
            self._classification_layer = keras.layers.Add()
            self._classification_layer.build(output_shapes)

    def call(self, inputs):
        x = []
        for i, input_vector in enumerate(inputs):
            x.append(self._lstm_blocks[i](input_vector))

        if not self._hard_coded_weights:
            x = self._reshape_layer(x)
            # x = self._scale_layer(x)

        return self._classification_layer(x)
    
    @keras.utils.register_keras_serializable()
    def model_loss_function(self, y_true, y_pred):
        return FatigueClassifier.loss(y_true, y_pred, self._mask)