from typing import Tuple, List
import numpy as np


class EllipseTrajectory:
  def __init__(self, semi_lengths: Tuple[float, float], center_coordinates: 
               Tuple[float, float], angle_init: float, angle_vel: float):
    self.semi_lengths = semi_lengths
    self.center = center_coordinates
    self.angle_init = angle_init
    self.angle_vel = angle_vel

  def get_linear_vel(self, t: float) -> List[float]:
    a, b = self.semi_lengths
    return [-a * np.cos(self.angle_vel * t + self.angle_init), 0.0, b * \
            np.sin(self.angle_vel * t + self.angle_init)]
  
  def get_self_rotation_vel(self, t: float) -> List[float]:
    return [0.0, self.angle_vel * t + self.angle_init - np.pi/2, 0.0]