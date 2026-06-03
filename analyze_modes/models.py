from dataclasses import dataclass


@dataclass
class SafetyThresholds:
    current_max: float
    current_min: float
    weight_max: float
    weight_min: float
    specific_current_max: float
    specific_current_min: float
    current_emergency: float
    weight_emergency: float
    specific_current_emergency: float

class KalmanFilter:
    def __init__(self, initial_state: float, process_variance: float = 1e-5, measurement_variance: float = 0.1 ** 2):
        self.state = initial_state
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate_error = 1.0

    def update(self, measurement: float) -> float:
        pred_state = self.state
        pred_error = self.estimate_error + self.process_variance

        kalman_gain = pred_error / (pred_error + self.measurement_variance)
        self.state = pred_state + kalman_gain * (measurement - pred_state)
        self.estimate_error = (1 - kalman_gain) * pred_error

        return self.state
