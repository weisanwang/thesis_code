import numpy as np

class ShiftDetector:
    def __init__(self,
                 slide_window_length: int,
                 mean_threshold: float,
                 var_threshold: float,
                 jump_threshold: float = 0.5):
        """
        slide_window_length: The length of slide window
        mean_threshold     : Threshold for the mean of the loss in the sliding window
        var_threshold      : Threshold for the variance of the loss in the sliding window
        jump_threshold     : Threshold for the jump of the loss between two batches
        """
        self.slide_window_length = slide_window_length
        self.mean_threshold = mean_threshold
        self.var_threshold = var_threshold
        self.jump_threshold = jump_threshold

        # Keep a sliding window of the last N loss values
        self.loss_window = []

        # Save the previous loss and current loss
        self.prev_batch_loss = None
        self.curr_batch_loss = None

        # Save the mean and variance of the last two windows
        self.prev_mean = None
        self.prev_var  = None

        # State flag: True indicates "Waiting for detecting plateau", False indicates "Wating for detecting new peak"
        self.new_peak_detected = True

        # Count the number of plateau detected
        self.plateau_count = 0
        # Count the number of peaks detected
        self.peak_count = 0

    def update(self, loss: float) -> bool:
        """
        Called after training each batch:
        - loss: The loss value (float) of the current batch Returns:
        - If a domain shift (new peak) is detected, returns True
        - Otherwise, returns False
        """
        # Append the current loss to the sliding window
        self.loss_window.append(loss)
        if len(self.loss_window) > self.slide_window_length:
            del self.loss_window[0]

        # If the sliding window is not full, return False
        if len(self.loss_window) < self.slide_window_length:
            return False

        # Store the batch loss
        self.prev_batch_loss = self.curr_batch_loss
        self.curr_batch_loss = loss

        # Calculate the mean and variance of the loss in the sliding window
        mean = float(np.mean(self.loss_window))
        var  = float(np.var(self.loss_window))

        # Detect the new peak (domain shift) only after the plateau and not the first window
        if (not self.new_peak_detected) and (self.prev_mean is not None) and (self.prev_batch_loss is not None):
            # Calculate the jump ratio
            jump_ratio = (self.curr_batch_loss - self.prev_batch_loss) / self.prev_batch_loss 
            # New peak detected if the mean of sliding window increases more than the standard deviation between the two windows
            if mean - self.prev_mean > np.sqrt(self.prev_var) and jump_ratio > self.jump_threshold:
                self.new_peak_detected = True
                self.peak_count += 1
                print(
                    f"[ShiftDetector] 🔺 Domain shift detected!"
                    f"mean={mean:.4f}, var={var:.6f}"
                    )
                # Save the mean and variance of current window
                self.prev_mean = mean
                self.prev_var  = var
                return True

        # Detect the plateau only after the new peak
        if self.new_peak_detected:
            if (mean < self.mean_threshold) and (var < self.var_threshold):
                self.plateau_count += 1
                print(
                    f"[ShiftDetector] ── Plateau #{self.plateau_count} detected!"
                    f"mean={mean:.4f}, var={var:.6f}"
                )
                # Switch to "Waiting for next peak" state
                self.new_peak_detected = False

        # Save the mean and variance of current window
        self.prev_mean = mean
        self.prev_var  = var

        return False
