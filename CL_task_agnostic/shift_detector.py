import numpy as np

class ShiftDetector:
    def __init__(self,
                 slide_window_length: int,
                 mean_threshold: float,
                 var_threshold: float,
                 jump_threshold: float):
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

        # Save the mean and variance of the last window
        self.prev_mean = None
        self.prev_var  = None
        # Save the mean and variance of the current window
        self.curr_mean = None
        self.curr_var  = None

        # State flag: True indicates "Waiting for detecting plateau", False indicates "Wating for detecting new peak"
        self.new_peak_detected = True

        # Count the number of plateau detected
        self.plateau_count = 0
        # Count the number of peaks detected
        self.peak_count = 0

    def update(self, loss: float) -> str:
        """
        Called after training each batch:
        - loss: The loss value (float) of the current batch Returns:
        - If a domain shift (new peak) is detected, returns True
        - Otherwise, returns False
        """
        # Append the current loss to the sliding window
        self.loss_window.append(loss)

        # If the sliding window is full, remove the oldest loss
        if len(self.loss_window) > self.slide_window_length:
            del self.loss_window[0]
        # If the sliding window is not full, return 'none'
        if len(self.loss_window) < self.slide_window_length:
            return 'none'

        # Store the batch loss
        self.prev_batch_loss = self.curr_batch_loss
        self.curr_batch_loss = loss

        # Calculate the mean and variance of the loss in the sliding window
        self.curr_mean = float(np.mean(self.loss_window))
        self.curr_var  = float(np.var(self.loss_window))

        status = 'none'
        if self.new_peak_detected:
            # plateau?
            if (self.curr_mean < self.mean_threshold) and (self.curr_var < self.var_threshold):
                status = 'plateau'
                self.new_peak_detected = False
                self.plateau_count += 1

        else:
            # peak?
            if (self.prev_mean is not None) and (self.prev_batch_loss is not None):
                # Calculate the jump ratio
                jump_ratio = (self.curr_batch_loss - self.prev_batch_loss) / self.prev_batch_loss 
                if (self.curr_mean - self.prev_mean > np.sqrt(self.prev_var)
                and jump_ratio > self.jump_threshold):
                    status = 'peak'
                    self.new_peak_detected = True
                    self.peak_count += 1
        
        # Update the previous mean and variance
        self.prev_mean, self.prev_var = self.curr_mean, self.curr_var
        
        return status
