import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation

def Rotation_fromOrientation(obj):
    return Rotation.from_euler('ZXZ', [ -obj.roll_deg,-obj.tilt_deg, obj.heading_deg ], degrees=True)
def Orientation_fromRotation(rr: Rotation):
    eul = rr.as_euler('ZXZ', True)
    return {"roll_deg":-eul[0], "tilt_deg":-eul[1], "heading_deg": eul[2]}

def px_to_proj_rays(pxls,fx, fy, cx, cy):
    r = np.array([-(pxls[:,0]-cx)/fx,(pxls[:,1]-cy)/fy,np.ones_like(pxls[:,1])]).T
    r /= np.linalg.norm(r, axis=-1)[..., None]
    return -r
def proj_coords_to_px(coords, fx, fy, cx, cy):
    return np.array([-coords[:,0]*fx/coords[:,2]+cx, coords[:,1]*fy/coords[:,2]+cy]).T
def world_to_pix(xyz, rr, fx,fy,cx,cy):
    projc=rr.apply(xyz)
    return proj_coords_to_px(projc, fx,fy,cx,cy)

def GetRotMatr(space_coords, pxls, fx, fy, cx, cy):
    proj_rays = px_to_proj_rays(pxls, fx , fy, cx, cy)
    rotmatr = Rotation.align_vectors(proj_rays,space_coords)[0]
    return rotmatr

def GetTestPxls(space_coords, pxls, fx, fy, cx, cy):
    rotmatr = GetRotMatr(space_coords, pxls, fx, fy, cx, cy)
    test_pxls = world_to_pix(space_coords, rotmatr, fx, fy, cx, cy)
    return test_pxls, rotmatr

def ResFOV(x, space_coords, pxls):
    test_pxls, rotmatr = GetTestPxls(space_coords, pxls, x[0], x[1], x[2], x[3])
    return np.sqrt(np.mean((test_pxls-pxls)**2))
def ResFOVCamera(x, camera, space_coords, pxls):
    """
    camera - cameratransform.camera
    """
    fx, fy, cx, cy = x[0], x[1], x[2], x[3]
    rotmatr = GetRotMatr(space_coords, pxls, fx, fy, cx, cy)
    camera.focallength_x_px = fx
    camera.focallength_y_px = fy
    camera.center_x_px=cx
    camera.center_y_px=cy
    angles = Orientation_fromRotation(rotmatr)
    camera.roll_deg=angles['roll_deg']
    camera.tilt_deg=angles['tilt_deg']
    camera.heading_deg=angles['heading_deg']
    if len(x)>4:
        camera.k1=x[4]
    test_pxls = camera.imageFromSpace(space_coords, hide_backpoints=False)
    return np.sqrt(np.mean((test_pxls-pxls)**2))
def reload_camera(camera):
    import cameratransform as ct
    keys = camera.parameters.parameters.keys()
    variables = {key: getattr(camera, key) for key in keys}
    camnew = ct.Camera(ct.RectilinearProjection(), ct.SpatialOrientation(),  ct.BrownLensDistortion())
    for key in variables:
        setattr(camnew, key, variables[key])
    return camnew
# fix loading bug of cameratransform v1.1
def load_camera(filename):
    import cameratransform as ct
    camera = ct.Camera(ct.RectilinearProjection(), ct.SpatialOrientation(),  ct.BrownLensDistortion())
    camera.load(filename)
    return camera

def OptimizeCamera(camera, enu_unit_coords, pxls, distortion=True):
    fx,fy, cx,cy =  camera.focallength_x_px, camera.focallength_y_px, camera.center_x_px, camera.center_y_px
    if distortion:
        optres = scipy.optimize.minimize(ResFOVCamera, [fx,fy, cx,cy, 0], args=(camera,  enu_unit_coords, pxls), method='SLSQP',
                        bounds=[[1000,10000], [1000,10000], [0,6000],[0,4000],[0,1]])
    else:
        optres = scipy.optimize.minimize(ResFOVCamera, [fx,fy, cx,cy], args=(camera,  enu_unit_coords, pxls), method='SLSQP',
                        bounds=[[1000,10000], [1000,10000], [0,6000],[0,4000]])
    print(optres)
    # construct new camera from optimized camera parameters
    cameranew = reload_camera(camera)
    return cameranew

def ResRot(x, camera1, space_coords, pxls):
    rr=Rotation.from_rotvec(x[0:3])
    #camera1.orientation.R = rr.as_matrix().T
    # atkodu kā leņķi definēti ImageTransform bibliotēkā
    # pašo ;eņķi slikti optimizējas
    # man nav ne jausmas, kāpēc šādi strādā, bet, ja šeit ieliek funkciju Orientation_fromRotation tad nē?
    aa = rr.as_euler('zxz', True)
    aa[2]=-aa[2]
    camera1.heading_deg=aa[0]
    camera1.tilt_deg=aa[1]
    camera1.roll_deg=aa[2]
    camera1.focallength_x_px=x[3]
    camera1.focallength_y_px=x[4]
    camera1.center_x_px=x[5]
    camera1.center_y_px=x[6]
    camera1.k1=x[7]
    return np.sqrt(np.mean((camera1.imageFromSpace(space_coords)-pxls)**2))
