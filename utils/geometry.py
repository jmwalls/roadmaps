"""Convenience functions and data structures to represent common geometric
relationships.
"""
import numpy as np
import transforms3d

def rot_from_rph_rad(r, p, h):
    """Generate rotation matrix from roll, pitch, heading specified in rads.
    XXX: comment on rotation order.
    """
    return transforms3d.euler.euler2mat(r, p, h)

def rot_from_rph_deg(r_deg, p_deg, h_deg):
    """Generate rotation matrix from roll, pitch, heading specified in degs.
    XXX: comment on rotation order.
    """
    return rot_from_rph_rad(*np.deg2rad([r_deg, p_deg, h_deg]))

class SE3:
    """Data structure to capture rigid body transformations--described by the
    SE(3) group.

    transform_AB = (R_AB, p_AB_A)

    Parameters
    ----------
    R : 3x3 rotation matrix
    p : 3, translation vector
    """
    def __init__(self, *, R=np.eye(3), p=np.zeros(3)):
        self.R = R
        self.p = p

    def inverse(self):
        """Inverse element.

        Returns
        -------
        SE3 : transform_BA = (R_AB.T, -R_AB.T @ p_AB_A)
        """
        return SE3(R=self.R.T, p=-(self.R.T @ self.p))

    def __matmul__(self, other):
        """Composes two SE3 objects.

        Returns
        -------
        SE3 : transform_AC = (R_AB @ R_BC, p_AB_A + R_AB @ p_BC_B)
              where transform_AB/transform_BC are self and other, respectively.
        """
        return SE3(R=self.R @ other.R, p=self.p + self.R @ other.p)

    def __mul__(self, p):
        """Represents the input point in self's frame.

        Returns
        -------
        p : 3, p_Ai_A = p_AB_A + R_AB p_Bi_B
        """
        return self.p + self.R @ p
