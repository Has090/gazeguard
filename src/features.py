"""Per-frame features from MediaPipe Face Mesh landmarks (478 points incl. iris).

Pure numpy/OpenCV so it can be unit-tested without MediaPipe.
"""
import cv2
import numpy as np

# Landmark indices (MediaPipe Face Mesh).
NOSE_TIP, CHIN = 1, 152
R_EYE_OUT, R_EYE_IN, L_EYE_IN, L_EYE_OUT = 33, 133, 362, 263   # subject's right / left eye
R_UP, R_LOW = (160, 158), (144, 153)
L_UP, L_LOW = (385, 387), (380, 373)
R_IRIS, L_IRIS = 468, 473
MOUTH_L, MOUTH_R, LIP_UP, LIP_LOW = 61, 291, 13, 14
PNP_IDX = [NOSE_TIP, CHIN, R_EYE_OUT, L_EYE_OUT, MOUTH_L, MOUTH_R]
# Generic 3D face model (mm) used with solvePnP.
MODEL_POINTS = np.array(
    [(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
     (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)],
    dtype=np.float64,
)

EMPTY = dict(face_present=0, yaw=0.0, pitch=0.0, roll=0.0, nose_x=0.0, nose_y=0.0,
             iris_l_x=0.0, iris_l_y=0.0, iris_r_x=0.0, iris_r_y=0.0, ear=0.0, mouth_open=0.0)


def _d(a, b):
    return float(np.linalg.norm(a - b))


def _wrap(angle):
    """Wrap to [-90, 90] (solvePnP with this model can come out flipped by 180 deg)."""
    angle = (angle + 180.0) % 360.0 - 180.0
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return float(angle)


def head_pose(px, w, h):
    """(yaw, pitch, roll) in degrees from 6 landmarks via solvePnP."""
    img_pts = px[PNP_IDX, :2].astype(np.float64)
    cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, img_pts, cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rot, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rot)[0]          # (pitch, yaw, roll) in degrees
    return _wrap(angles[1]), _wrap(angles[0]), _wrap(angles[2])


def eye_aspect_ratio(px):
    r = (_d(px[R_UP[0], :2], px[R_LOW[0], :2]) + _d(px[R_UP[1], :2], px[R_LOW[1], :2])) / (
        2.0 * _d(px[R_EYE_OUT, :2], px[R_EYE_IN, :2]) + 1e-9)
    l = (_d(px[L_UP[0], :2], px[L_LOW[0], :2]) + _d(px[L_UP[1], :2], px[L_LOW[1], :2])) / (
        2.0 * _d(px[L_EYE_IN, :2], px[L_EYE_OUT, :2]) + 1e-9)
    return (r + l) / 2.0


def iris_offset(px, iris, c1, c2):
    """Iris centre relative to the eye centre, divided by eye width."""
    centre = (px[c1, :2] + px[c2, :2]) / 2.0
    width = _d(px[c1, :2], px[c2, :2]) + 1e-9
    off = (px[iris, :2] - centre) / width
    return float(off[0]), float(off[1])


def landmarks_to_features(norm_xy, w, h):
    """norm_xy: (N>=478, 2+) array of normalised landmark coords. Returns a feature dict."""
    px = np.asarray(norm_xy, dtype=np.float64)[:, :2] * np.array([w, h], dtype=np.float64)
    if px.shape[0] < 478:
        return dict(EMPTY)
    xmin, ymin = px.min(axis=0)
    xmax, ymax = px.max(axis=0)
    fw, fh = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    yaw, pitch, roll = head_pose(px, w, h)
    il = iris_offset(px, L_IRIS, L_EYE_IN, L_EYE_OUT)
    ir = iris_offset(px, R_IRIS, R_EYE_OUT, R_EYE_IN)
    mouth = _d(px[LIP_UP, :2], px[LIP_LOW, :2]) / (_d(px[MOUTH_L, :2], px[MOUTH_R, :2]) + 1e-9)
    return dict(
        face_present=1, yaw=yaw, pitch=pitch, roll=roll,
        nose_x=float((px[NOSE_TIP, 0] - xmin) / fw), nose_y=float((px[NOSE_TIP, 1] - ymin) / fh),
        iris_l_x=il[0], iris_l_y=il[1], iris_r_x=ir[0], iris_r_y=ir[1],
        ear=float(eye_aspect_ratio(px)), mouth_open=float(mouth),
    )
