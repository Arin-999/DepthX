
import math


class ExponentialSmoother:
    def __init__(self, alpha=0.3):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")

        self.alpha = alpha
        self.value = None

    def update(self, measurement):
        if measurement is None:
            return self.value

        if not math.isfinite(measurement):
            return self.value

        measurement = float(measurement)

        if self.value is None:
            self.value = measurement
        else:
            self.value = (
                self.alpha * measurement
                + (1 - self.alpha) * self.value
            )

        return self.value

    def reset(self):
        self.value = None
