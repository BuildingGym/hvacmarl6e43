
class PVPredictorModel(Sequential):
    @staticmethod
    def _invert_normalizer(normalizer: layers.Normalization):
        return layers.Normalization(
            axis=normalizer.axis, 
            invert=True,
            **(
                dict(
                    mean=normalizer.mean, 
                    variance=normalizer.variance,                 
                ) 
                if normalizer.built else 
                dict()
            ),
        )
    
    def __init__(self, **kwargs):
        super().__init__([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.5),
            layers.Bidirectional(layers.LSTM(100, return_sequences=True)),
            layers.Dropout(0.5),
            layers.Bidirectional(layers.LSTM(100, return_sequences=False)),
            layers.Dropout(0.5),
            layers.Dense(1),
        ], **kwargs)
        self._input_normalizer = layers.Normalization(axis=-1)
        self._output_normalizer = layers.Normalization(axis=-1)

    def compile(self, **kwargs):
        return super().compile(optimizer='adam', loss='mse', run_eagerly=True, jit_compile=False, **kwargs)

    def fit(self, inputs, outputs, **kwargs):
        self._input_normalizer.adapt(inputs)
        self._output_normalizer.adapt(outputs)

        return super().fit(
            x=self._input_normalizer(inputs),
            y=self._output_normalizer(outputs),
            **kwargs,
        )
    
    def predict(self, inputs, **kwargs): 
        output = super().predict(
            x=self._input_normalizer(inputs),
            **kwargs,
        )

        denormalizer = self._invert_normalizer(
            self._output_normalizer
        )
        return denormalizer(output)



import keras 

class PVPredictorModel(Model):
    @staticmethod
    def _invert_normalizer(normalizer: layers.Normalization):
        return layers.Normalization(
            axis=normalizer.axis, 
            invert=True,
            **(
                dict(
                    mean=normalizer.mean, 
                    variance=normalizer.variance,                 
                ) 
                if normalizer.built else 
                dict()
            ),
        )

    def __init__(self):
        super().__init__()
        self._body = layers.Pipeline([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.5),
            layers.Bidirectional(layers.LSTM(100, return_sequences=True)),
            layers.Dropout(0.5),
            layers.Bidirectional(layers.LSTM(100, return_sequences=False)),
            layers.Dropout(0.5),
            layers.Dense(1),
        ])
        self._input_normalizer = layers.Normalization(axis=-1)
        self._output_normalizer = layers.Normalization(axis=-1)

    def compile(self, **kwargs):
        return super().compile(
            optimizer='adam', loss='mse',
            run_eagerly=True, jit_compile=False, 
            **kwargs,
        )
    
    def fit(self, inputs, outputs, **kwargs):
        self._input_normalizer.adapt(inputs)
        self._output_normalizer.adapt(outputs)

        return super().fit(
            x=self._input_normalizer(inputs),
            y=self._output_normalizer(outputs),
            **kwargs,
        )
    
    def predict(self, inputs, **kwargs): 
        output = super().predict(
            x=self._input_normalizer(inputs),
            **kwargs,
        )

        denormalizer = self._invert_normalizer(
            self._output_normalizer
        )
        return denormalizer(output)

    def call(self, inputs):
        return self._body(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "_body": keras.layers.serialize(self._body),
            "_input_normalizer": keras.layers.serialize(self._input_normalizer),
            "_output_normalizer": keras.layers.serialize(self._output_normalizer),
        })
        return config

    @classmethod
    def from_config(cls, config):
        body = keras.layers.deserialize(config.pop("_body"))
        input_normalizer = keras.layers.deserialize(config.pop("_input_normalizer"))
        output_normalizer = keras.layers.deserialize(config.pop("_output_normalizer"))

        instance = cls()
        instance._body = body
        instance._input_normalizer = input_normalizer
        instance._output_normalizer = output_normalizer

        return instance




# class Preprocessor(Model):
#     def __init__(self):
#         super().__init__()
#         self._normalizer = layers.Normalization(axis=-1)

#     def adapt(self, x):
#         self._normalizer.adapt(x)
#         return self

#     def transform(self, x):
#         return self._normalizer(x)
    
#     @staticmethod
#     def _invert_normalizer(normalizer: layers.Normalization):
#         return layers.Normalization(
#             axis=normalizer.axis, 
#             invert=True,
#             **(
#                 dict(
#                     mean=normalizer.mean, 
#                     variance=normalizer.variance,                 
#                 ) 
#                 if normalizer.built else 
#                 dict()
#             ),
#         )
    
#     def inverse_transform(self, x):
#         return self._invert_normalizer(self._normalizer)(x)
        
#     def call(self, x, training=False):
#         if training:
#             return self.adapt(x)
#         return self.transform(x)



# class PVPredictorModel:
#     @staticmethod
#     def _invert_normalizer(normalizer: layers.Normalization):
#         return layers.Normalization(
#             axis=normalizer.axis, 
#             invert=True,
#             **(
#                 dict(
#                     mean=normalizer.mean, 
#                     variance=normalizer.variance,                 
#                 ) 
#                 if normalizer.built else 
#                 dict()
#             ),
#         )

#     @_functools_.cached_property
#     def _input_normalizer(self):
#         return layers.Normalization(
#             axis=-1,
#         )
    
#     @_functools_.cached_property
#     def _output_normalizer(self):
#         return layers.Normalization(
#             axis=-1,
#         )
    
#     @_functools_.cached_property
#     def _model(self):
#         model = Sequential([
#             layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
#             layers.MaxPooling1D(pool_size=2),
#             layers.Dropout(0.5),
#             layers.Bidirectional(layers.LSTM(100, return_sequences=True)),
#             layers.Dropout(0.5),
#             layers.Bidirectional(layers.LSTM(100, return_sequences=False)),
#             layers.Dropout(0.5),
#             layers.Dense(1),
#         ])
#         # TODO customize
#         model.compile(optimizer='adam', loss='mse', run_eagerly=True, jit_compile=False)
#         return model
    
#     def fit(self, inputs, outputs, **kwargs):
#         self._input_normalizer(inputs, training=True)
#         self._output_normalizer(outputs, training=True)

#         return self._model.fit(
#             x=self._input_normalizer(inputs),
#             y=self._output_normalizer(outputs),
#             **kwargs,
#         )
    
#     def predict(self, inputs, **kwargs): 
#         output = self._model.predict(
#             x=self._input_normalizer(inputs),
#             **kwargs,
#         )

#         denormalizer = self._invert_normalizer(
#             self._output_normalizer
#         )
#         return denormalizer(output)


# class PVPredictorModel:
#     @_functools_.cached_property
#     def _input_preproc(self):
#         preproc = Preprocessor()
#         preproc.compile(run_eagerly=True)
#         return preproc
    
#     @_functools_.cached_property
#     def _output_preproc(self):
#         preproc = Preprocessor()
#         preproc.compile(run_eagerly=True)
#         return preproc
    
#     @_functools_.cached_property
#     def _model(self):
#         model = Sequential([
#             layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
#             layers.MaxPooling1D(pool_size=2),
#             layers.Dropout(0.5),
#             layers.Bidirectional(layers.LSTM(100, return_sequences=True)),
#             layers.Dropout(0.5),
#             layers.Bidirectional(layers.LSTM(100, return_sequences=False)),
#             layers.Dropout(0.5),
#             layers.Dense(1),
#         ])
#         # TODO customize
#         model.compile(optimizer='adam', loss='mse', run_eagerly=True, jit_compile=False)
#         return model
    
#     def fit(self, inputs, outputs, **kwargs):
#         # return self._model.fit(
#         #     x=inputs,
#         #     y=outputs,
#         #     **kwargs,
#         # )

#         self._input_preproc.adapt(inputs)
#         self._output_preproc.adapt(outputs)

#         return self._model.fit(
#             x=self._input_preproc(inputs),
#             y=self._output_preproc(outputs),
#             **kwargs,
#         )
    
#     def predict(self, inputs, **kwargs): 
#         #return self._model.predict(inputs, **kwargs)

#         output = self._model.predict(
#             x=self._input_preproc(inputs),
#             **kwargs,
#         )

#         return self._output_preproc.inverse_transform(output)



import numpy as np
import plotly.graph_objs as go

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

df = df_multi_agent_a_0_4[['elec', 'pmv']]
df['elec_norm'] = df['elec'] / (df['elec'].max())
df['pmv_norm_abs'] = df['pmv'].abs() / (df['pmv'].abs().max())

theta, rho = cart2pol(df['pmv_norm_abs'], df['elec_norm'])
fig = plotly.graph_objs.Figure(
    data=[
        go.Scatterpolar(
            r=rho,
            theta=theta,
            mode='markers',
        )
    ],
    layout=go.Layout(
        xaxis_title="elec",
        yaxis_title="pmv_abs"
    )
)
fig
