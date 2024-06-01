import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation
import cameraprojections

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
def ResFOVCamera(x, camera, space_coords, pxls, distortion=True, centers=True, separate_x_y=True):
    """
    camera - cameratransform.camera
    """
    if separate_x_y:
        fx, fy = x[0], x[1]
        n=2
    else:
        fx=fy=x[0]
        n=1

    if centers:
        n=n+2
        cx, cy = x[2], x[3]
    else:
        cx, cy = camera.center_x_px, camera.center_y_px
    rotmatr = GetRotMatr(space_coords, pxls, fx, fy, cx, cy)
    camera.focallength_x_px = fx
    camera.focallength_y_px = fy
    camera.center_x_px=cx
    camera.center_y_px=cy
    angles = Orientation_fromRotation(rotmatr)
    camera.roll_deg=angles['roll_deg']
    camera.tilt_deg=angles['tilt_deg']
    camera.heading_deg=angles['heading_deg']
    if distortion:
        camera.k1=x[n]
    test_pxls = camera.imageFromSpace(space_coords, hide_backpoints=False)
    return np.sqrt(np.mean((test_pxls-pxls)**2))
def reload_camera(camera):
    import cameratransform as ct
    keys = camera.parameters.parameters.keys()
    variables = {key: getattr(camera, key) for key in keys}
    camnew = ct.Camera(type(camera.projection)(), ct.SpatialOrientation(),  ct.BrownLensDistortion())
    for key in variables:
        setattr(camnew, key, variables[key])
    return camnew
# fix loading bug of cameratransform v1.1
def load_camera(filename):
    import cameratransform as ct
    import json
    with open(filename, "r") as fp:
        variables = json.loads(fp.read())
    if 'projectiontype' in variables:
        projection = cameraprojections.projection_by_name(variables['projectiontype'])
    else:
        projection = ct.RectilinearProjection

    camera = ct.Camera(projection(), ct.SpatialOrientation(),  ct.BrownLensDistortion())    
    camera.load(filename)
    return camera
def save_camera(camera, filename):
    import cameratransform as ct
    import json
    keys = camera.parameters.parameters.keys()
    export_dict = {key: getattr(camera, key) for key in keys}
    export_dict['projectiontype']= cameraprojections.name_by_projection(camera.projection)
    with open(filename, "w") as fp:
        fp.write(json.dumps(export_dict, indent=4))

def OptimizeCamera(camera, enu_unit_coords, pxls, distortion=True, centers=True,  separate_x_y=True,
                   f_bounds=[1000,10000], cx_bounds=[0, 6000], cy_bounds=[0,4000]):
    fx,fy, cx,cy =  camera.focallength_x_px, camera.focallength_y_px, camera.center_x_px, camera.center_y_px
    #print(f_bounds, cx_bounds, cy_bounds)
    x0 = [fx]

    bounds=[f_bounds]
    if separate_x_y:
        bounds=bounds+[f_bounds]
        x0 = x0 + [fy]
    if centers:
        x0=x0+[cx,cy]
        bounds = bounds + [cx_bounds, cy_bounds]
    if distortion:
        x0=x0+[0.0]
        bounds = bounds + [[0.0, 1.0]]


    optres = scipy.optimize.minimize(ResFOVCamera, x0, args=(camera,  enu_unit_coords, pxls, distortion, centers,separate_x_y), method='SLSQP',
                        bounds=bounds)
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
