"""Utilities for loading data from publicly available data sets.
"""
import glob
import os
from shutil import rmtree
import urllib.request
import zipfile

import numpy as np
import pandas as pd

import pymap3d as pm  # Only necessary for LLA to ENU conversion.


KITTI_URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data'
KITTI_OUT = 'data-cache'


OXTS_COLUMNS = [
    'lat',           # latitude of the oxts-unit (deg)
    'lon',           # longitude of the oxts-unit (deg)
    'alt',           # altitude of the oxts-unit (m)
    'roll',          # roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
    'pitch',         # pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
    'yaw',           # heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
    'vn',            # velocity towards north (m/s)
    've',            # velocity towards east (m/s)
    'vf',            # forward velocity, i.e. parallel to earth-surface (m/s)
    'vl',            # leftward velocity, i.e. parallel to earth-surface (m/s)
    'vu',            # upward velocity, i.e. perpendicular to earth-surface (m/s)
    'ax',            # acceleration in x, i.e. in direction of vehicle front (m/s^2)
    'ay',            # acceleration in y, i.e. in direction of vehicle left (m/s^2)
    'az',            # acceleration in z, i.e. in direction of vehicle top (m/s^2)
    'af',            # forward acceleration (m/s^2)
    'al',            # leftward acceleration (m/s^2)
    'au',            # upward acceleration (m/s^2)
    'wx',            # angular rate around x (rad/s)
    'wy',            # angular rate around y (rad/s)
    'wz',            # angular rate around z (rad/s)
    'wf',            # angular rate around forward axis (rad/s)
    'wl',            # angular rate around leftward axis (rad/s)
    'wu',            # angular rate around upward axis (rad/s)
    'pos_accuracy',  # velocity accuracy (north/east in m)
    'vel_accuracy',  # velocity accuracy (north/east in m/s)
    'navstat',       # navigation status (see navstat_to_string)
    'numsats',       # number of satellites tracked by primary GPS receiver
    'posmode',       # position mode of primary GPS receiver (see gps_mode_to_string)
    'velmode',       # velocity mode of primary GPS receiver (see gps_mode_to_string)
    'orimode']       # orientation mode of primary GPS receiver (see gps_mode_to_string)


def _parse_timestamp_file(path):
    assert os.path.exists(path)
    with open(path, 'r') as f:
        datestr = [l for l in f.readlines()]
    return datestr


def _load_kitti_oxts(*, path):
    # Load data one entry at a time...
    files = sorted([f for f in glob.glob(os.path.join(path, 'data/*.txt'))])
    data = np.array([np.fromfile(f, sep=' ') for f in files])

    # Corresponding timestamps are in another file...
    index = pd.DatetimeIndex(
            data=_parse_timestamp_file(os.path.join(path, 'timestamps.txt')))
    assert len(index) == data.shape[0]

    # Create data frame ane augment with timestamp seconds column.
    df = pd.DataFrame(data, index=index, columns=OXTS_COLUMNS)
    df['timestamp'] = 1e-9 * index.astype(np.int64)
    return df


def _load_kitti_image(*, base, index):
    index_str = f'{index:02d}'
    index = pd.DatetimeIndex(
            data=_parse_timestamp_file(os.path.join(base, f'image_{index_str}', 'timestamps.txt')))
    paths = sorted([f for f in glob.glob(os.path.join(base, f'image_{index_str}', 'data/*.png'))])
    return pd.DataFrame(data={f'timestamp_{index_str}': (1e-9 * index.astype(np.int64)),
                              f'image_{index_str}': paths})


def _parse_calib(path):
    with open(path, 'r') as f:
        calib = {p[0] : p[2].strip()
                    for p in list(map(lambda l : l.partition(':'),
                                      [l for l in f.readlines()]))}
    return calib


class Calibration:
    """Represent KITTI calibration parameters.

    Refer to [KITTI data description](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf)

    Parameters
    ----------
    P_rect_00 : XXX
    R_rect_00 : XXX

    R_CV : rotation component of rigid body transformation between velodyne (V)
           and camera (C) frame.
    p_CV_C  : translation component of rigid body transformation between
              velodyne (V) and camera (C) frame.

    R_VI : rotation component of rigid body transformation between velodyne (V)
           and camera (C) frame.
    p_VI_V  : translation component of rigid body transformation between
              velodyne (V) and camera (C) frame.

    TODO: add intrinsics/extrinsics for cameras 1/2/3
    """
    def __init__(self):
        self.P_rect_00 = None
        self.R_rect_00 = None

        self.R_CV = None
        self.p_CV_C = None

        self.R_VI = None
        self.p_VI_V = None

    @staticmethod
    def create_from_data(*, path):
        """Parse calibrations from extracted data.

        Parameters
        ----------
        path : path to extracted drive/sequence data directory containing
        calibration files.
        """
        calib = Calibration()

        # Read camera intrinsics.
        params = _parse_calib(os.path.join(path, 'calib_cam_to_cam.txt'))
        calib.P_rect_00 = np.fromstring(params['P_rect_00'], sep=' ').reshape(3, 4)
        calib.R_rect_00 = np.fromstring(params['R_rect_00'], sep=' ').reshape(3, 3)

        # Read camera/velodyne extrinsics.
        params = _parse_calib(os.path.join(path, 'calib_velo_to_cam.txt'))
        calib.R_CV = np.fromstring(params['R'], sep=' ').reshape(3, 3)
        calib.p_CV_C = np.fromstring(params['T'], sep=' ')

        # Read velodyne/imu extrinsics.
        params = _parse_calib(os.path.join(path, 'calib_imu_to_velo.txt'))
        calib.R_VI = np.fromstring(params['R'], sep=' ').reshape(3, 3)
        calib.p_VI_V = np.fromstring(params['T'], sep=' ')

        return calib


def get_kitti_data(*, drive, sequence, force_download=False):
    """
    Download raw odometry data from KITTI.

    Note: we download the raw unsynced data since this includes 100Hz OXTS
    outputs.

    Parameters
    ----------
    drive : (str) drive string, e.g., 2011_09_26
    sequence : (int) drive sequence number, e.g., 1
    force_download : (bool) optionally require new download

    Returns
    -------
    calibration
    df :  pandas dataframe
    """
    os.makedirs(KITTI_OUT, exist_ok=True)
    path = os.path.join(KITTI_OUT, drive)
    if force_download or not os.path.exists(path):
        def _download_extract(fin):
            url = f'{KITTI_URL}/{fin}'
            print(f'downloading {url}...')
            fout, _ = urllib.request.urlretrieve(url)  # downloads to tmp
            print(f'extracting {fout}...')
            with zipfile.ZipFile(fout, 'r') as z:  # extracts to pwd
                z.extractall(path=KITTI_OUT)
        _download_extract(f'{drive}_calib.zip')
        _download_extract(f'{drive}_drive_{sequence:04d}/{drive}_drive_{sequence:04d}_sync.zip')

        # Remove velodyne data since we only care about images/oxts.
        rmtree(os.path.join(path, f'{drive}_drive_{sequence:04d}_sync', 'velodyne_points'))
    assert os.path.exists(path)

    # Construct data frame from OXTS.
    df_oxts = _load_kitti_oxts(
        path=os.path.join(path, f'{drive}_drive_{sequence:04d}_sync', 'oxts'))

    # Add columns for ENU position.
    enu = pm.geodetic2enu(df_oxts['lat'], df_oxts['lon'], df_oxts['alt'],
                          df_oxts['lat'][0], df_oxts['lon'][0], df_oxts['alt'][0],
                          ell=pm.utils.Ellipsoid('wgs84'),
                          deg=True)
    df_oxts['pe'] = enu[0]
    df_oxts['pn'] = enu[1]
    df_oxts['pu'] = enu[2]

    # Construct list of data frames from each camera--should have 4 cameras.
    dfs = [_load_kitti_image(base=os.path.join(path, f'{drive}_drive_{sequence:04d}_sync'),
                             index=i) for i in range(4)]
    df = pd.concat([df_oxts, *[d.set_index(df_oxts.index) for d in dfs]], axis=1)

    # Load intrinsics calibrations.
    calibration = Calibration.create_from_data(path=path)

    # Add data from each camera--timestamps and relative paths to images.
    return calibration, df
