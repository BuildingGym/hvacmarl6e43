from typing import TypedDict, Unpack
import functools as _functools_

from more_itertools import sliding_window 
from keras import Model, Sequential, layers, KerasTensor

import numpy as np
import sklearn.preprocessing


class BatchStandardScaler:
    def __init__(self):
        self._scaler = sklearn.preprocessing.StandardScaler()

    def _flatten(self, x):
        x_ = np.array(np.concatenate(x, axis=0))
        if x_.ndim <= 1:
            return x_.reshape(-1, 1)
        return x_

    def fit(self, x):
        self._scaler.fit(self._flatten(x))
        return self

    def transform(self, x):
        return np.reshape(
            self._scaler.transform(self._flatten(x)),
            np.shape(x),
        )

    def inverse_transform(self, x):
        return np.reshape(
            self._scaler.inverse_transform(
                self._flatten(x)
            ),
            np.shape(x),
        )

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class PVPredictorModel:
    @_functools_.cached_property
    def _input_preproc(self):
        return BatchStandardScaler()
    
    @_functools_.cached_property
    def _output_preproc(self):
        return BatchStandardScaler()
    
    @_functools_.cached_property
    def _model(self):
        model = Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.5),
            layers.Bidirectional(layers.LSTM(100, return_sequences=True)),
            layers.Dropout(0.5),
            layers.Bidirectional(layers.LSTM(100, return_sequences=False)),
            layers.Dropout(0.5),
            layers.Dense(1),
        ])
        # TODO customize
        model.compile(optimizer='adam', loss='mse', run_eagerly=True, jit_compile=False)
        return model
    
    def fit(self, inputs, outputs, **kwargs):
        # return self._model.fit(
        #     x=inputs,
        #     y=outputs,
        #     **kwargs,
        # )

        return self._model.fit(
            x=self._input_preproc.fit_transform(inputs), 
            y=self._output_preproc.fit_transform(outputs), 
            **kwargs,
        )
    
    def predict(self, inputs, **kwargs): 
        #return self._model.predict(inputs, **kwargs)

        output = self._model.predict(
            x=self._input_preproc.transform(inputs),
            **kwargs,
        )

        return self._output_preproc.inverse_transform(output)



from collections import deque

class PVPredictor:
    class Observation(TypedDict):
        out_w: float
        r"""The current photovoltaic power output (W)"""

        @classmethod
        def to_array(cls, obs):
            if isinstance(obs, dict):
                return np.array(list(obs.values()))
            return np.array([cls.to_array(x) for x in obs])

    class Result(TypedDict):
        out_w: float
        r"""The next photovoltaic power output (W)"""

    Batch = list

    Truth = tuple[list[Observation], Result]

    def __init__(self, lookback: int | None = None):
        self.model = PVPredictorModel()
        self.observations = deque[self.Observation](maxlen=lookback)
        self._built = False
    
    def fit(self, data_batch: Batch[Truth], **kwargs):
        if not self._built:
            # TODO
            #self.model.compile(run_eagerly=True, jit_compile=False, )
            pass

        inputs = np.array([
            [list(obs.values()) for obs in obss] 
            for obss, _ in data_batch
        ])
        outputs = np.array([
            list(res.values()) 
            for _, res in data_batch
        ])

        res = self.model.fit(
            inputs, outputs,
            shuffle=False,
            **kwargs,
        )

        self._built = True

        return res

    def predict(self, data_batch: Batch[list[Observation]], **kwargs) -> Batch[Result]:
        output_batch = self.model.predict(
            np.array([[list(obs.values()) for obs in batch] for batch in data_batch]),
            **kwargs,
        )
        res = [{'out_w': output[0]} for output in output_batch]
        return res

    @_functools_.cached_property
    def experience(self) -> list[Truth]:
        return list()
    
    @property
    def true_experience(self) -> list[Truth]:
        return [
            (obs_curr, {'out_w': obs_next[0]['out_w']})
            for (obs_curr, _), (obs_next, _) in 
            sliding_window(self.experience, n=2)
        ]
    
    def __call__(
        self, 
        observation: Observation,
        lookback: int | None = None,
        experience_len: int = 2,
        fit_options: dict = dict(),
        predict_options: dict = dict(),
    ) -> Result:
        self.observations.append(observation)
        if len(self.observations) < (
            lookback 
            if lookback is not None else 
            # TODO what if this is None as well??
            self.observations.maxlen
        ):
            return None
            
        self.experience.append((list(self.observations), None))
        if len(self.experience) >= experience_len:
            self.fit(list(self.true_experience), **fit_options)
            self.experience.clear()
        
        if not self._built:
            return None
        
        [res, ] = self.predict([list(self.observations)], **predict_options)
        return res
