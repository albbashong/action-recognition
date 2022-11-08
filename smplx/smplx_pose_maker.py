import os.path as osp
import argparse
import numpy as np
import torch
import threading
import smplx
import time
import pyrender

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P

def main(model_folder,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=False,
         num_betas=10,
         sample_shape=True,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False):

    #body_pose generator
    body_pose = torch.rand(1, 63)




    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)

    betas, expression = None, None
    if sample_shape:
        betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    if sample_expression:
        expression = torch.randn(
            [1, model.num_expression_coeffs], dtype=torch.float32)

    output = model(betas=betas, expression=expression,
                   return_verts=True,body_pose=body_pose)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()
    print(joints)


    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    if plotting_module == 'pyrender':
        import pyrender
        import trimesh
        import matplotlib.pyplot as plt
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, model.faces,
                                   vertex_colors=vertex_colors)


        # camera add

        # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        s = np.sqrt(2) / 2
        camera_pose = np.array([
        [0.0, -s, s, 0.3],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, s, s, 0.35],
        [0.0, 0.0, 0.0, 1.0],
        ])

        human_pose = np.eye(4)
        #human_pose[:3,1]=np.random.randn(1,3)

        camera = WeakPerspectiveCamera(
            scale=[0.5, 0.5],
            translation=[0.5, 0.5],
            zfar=1000.,

        )

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        node_mesh = pyrender.Node(mesh=mesh, matrix=human_pose)
        node_camera = pyrender.Node(camera=camera, matrix=np.eye(4))

        light=pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                                 innerConeAngle=np.pi/16.0,
                                 outerConeAngle=np.pi/6.0)


        scene = pyrender.Scene()
        scene.add_node(node_mesh)
        scene.add_node(node_camera)
        #scene.add(light,pose=camera_pose)
        #scene.add(camera,pose=camera_pose)
        # if plot_joints:
        #     sm = trimesh.creation.uv_sphere(radius=0.005)
        #     sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        #     tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        #     tfs[:, :3, 3] = joints
        #     joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        #     scene.add(joints_pcl,pose=human_pose)

        py_viewer = pyrender.Viewer(scene, use_raymond_lighting=True,
                                    run_in_thread=True)
        i=0
        while py_viewer.is_active:
            human_pose[:3, 3] =[0, 0, -1]

            camera_pose=human_pose
            py_viewer.render_lock.acquire()
            scene.set_pose(node_mesh,human_pose)
            scene.set_pose(node_camera,camera_pose)
            py_viewer.render_lock.release()
            time.sleep(0.1)
            i+=0.001
        image = pyrender.OffscreenRenderer(400, 400)
        color, depth = image.render(scene)
        plt.imshow(color)
        plt.show()
        plt.pause(1)
        plt.close()




if __name__ == '__main__':

    model_folder = "./model"
    model_type = "smplx"
    plot_joints = True
    use_face_contour = False
    gender = "neutral"
    ext = 'npz'
    plotting_module = 'pyrender'
    num_betas = 10
    num_expression_coeffs = 10
    sample_shape = True
    sample_expression = True

    main(model_folder, model_type, ext=ext,
         gender=gender, plot_joints=plot_joints,
         num_betas=num_betas,
         num_expression_coeffs=num_expression_coeffs,
         sample_shape=sample_shape,
         sample_expression=sample_expression,
         plotting_module=plotting_module,
         use_face_contour=use_face_contour
         )

