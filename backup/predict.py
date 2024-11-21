from typing import TypedDict, Unpack
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

class predict:
    class Input(TypedDict):
        PV_W: float
        r"""The current photovoltaic power output (W)"""
        Drybulb_Temperature: float
        Air_Pressure: float
        Wind_Speed: float
        Wind_Direction: float
        Diffuse_Solar_Radiation: float
        Direct_Solar_Radiation: float
        Solar_Azimuth_Angle: float
        Solar_Altitude_Angle: float

    class Output(TypedDict):
        PV_W: float
        r"""The next photovoltaic power output (W)"""

    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = self.create_model((20, 9))  # Assuming 20 timesteps, 9 features
        self.X_buffer = []
        self.y_buffer = []

    def create_model(self, input_shape):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(100, return_sequences=False)))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def update(self, input: Input, target: float):
        input_data = [input[key] for key in input]
        self.X_buffer.append(input_data)
        self.y_buffer.append(target)
        if len(self.X_buffer) > 20:  # Maintain sequence length of 20
            self.X_buffer.pop(0)
            self.y_buffer.pop(0)

        if len(self.X_buffer) == 20:
            X_scaled = self.scaler_X.fit_transform(self.X_buffer)
            y_scaled = self.scaler_y.fit_transform(np.array(self.y_buffer).reshape(-1, 1))
            X_seq = np.expand_dims(X_scaled, axis=0)
            y_seq = np.expand_dims(y_scaled[-1], axis=0)

            self.model.fit(X_seq, y_seq, epochs=1, batch_size=1, verbose=0)

    def __call__(self, input: Input, **input_kwds: Unpack[Input]) -> Output:
        """TODO"""

        input = self.Input(input, **input_kwds)

        input_data = [input[f] for f in input]
        input_scaled = self.scaler_X.transform([input_data])
        input_seq = np.expand_dims(input_scaled, axis=0)

        prediction_scaled = self.model(input_seq, training=False).numpy()
        prediction = self.scaler_y.inverse_transform(prediction_scaled)

        return self.Output(PV_W=prediction[0][0])
        
