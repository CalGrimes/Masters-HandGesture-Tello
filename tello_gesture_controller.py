from djitellopy import Tello


class TelloGestureController:
    def __init__(self, tello: Tello):
        self.tello = tello
        self._is_landing = False

        # RC control velocities
        self.forw_back_velocity = 0
        self.up_down_velocity = 0
        self.left_right_velocity = 0
        self.yaw_velocity = 0

    def gesture_control(self, gesture_buffer):
        gesture_id = gesture_buffer.get_gesture()
        print("GESTURE", gesture_id)

        if not self._is_landing:
            if gesture_id == 0:  # Forward
                self.forw_back_velocity = 30
            elif gesture_id == 1:  # STOP
                self.forw_back_velocity = self.up_down_velocity = \
                    self.left_right_velocity = self.yaw_velocity = 0
            if gesture_id == 2:  # Back
                self.forw_back_velocity = -30

            elif gesture_id == -1:
                self.forw_back_velocity = self.up_down_velocity = \
                    self.left_right_velocity = self.yaw_velocity = 0

            self.tello.send_rc_control(self.left_right_velocity, self.forw_back_velocity,
                                       self.up_down_velocity, self.yaw_velocity)