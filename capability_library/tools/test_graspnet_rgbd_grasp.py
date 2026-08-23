import numpy as np

from graspnet_rgbd_grasp import model_free_collision_metrics


def _raw_grasp():
    row=np.zeros((1,17),dtype=float)
    row[0,1]=.04; row[0,2]=.02; row[0,3]=.02
    row[0,4:13]=np.eye(3).reshape(-1)
    return row


def test_model_free_collision_detects_finger_occupancy():
    clear=np.array([[0.,0.,0.]])
    blocked=np.array([[x,-.026,z] for x in np.linspace(-.035,.015,8) for z in np.linspace(-.008,.008,4)])
    clear_collision,_=model_free_collision_metrics(_raw_grasp(),clear)
    blocked_collision,_=model_free_collision_metrics(_raw_grasp(),blocked)
    assert blocked_collision[0] > clear_collision[0]


def test_model_free_collision_reports_inner_occupancy():
    inside=np.array([[x,0.,0.] for x in np.linspace(-.03,.015,12)])
    _,occupancy=model_free_collision_metrics(_raw_grasp(),inside)
    assert occupancy[0] > 0
