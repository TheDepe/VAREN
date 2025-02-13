from typing import Optional, Dict, Union
import os
import os.path as osp

import pickle

import numpy as np

import torch
import torch.nn as nn

from .lbs import (
    lbs, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords, blend_shapes)

from .vertex_ids import vertex_ids as VERTEX_IDS
from .vertex_ids import poll_vert_ids as POLL_VERT_IDS

from .utils import (
    Struct, to_np, to_tensor, Tensor, Array,
    SMALOutput, VARENOutput, MuscleDeformer, axis_angle_to_quaternion)
from .vertex_joint_selector import VertexJointSelector
from collections import namedtuple

TensorOutput = namedtuple('TensorOutput',
                          ['vertices', 'joints', 'betas', 'expression', 'global_orient', 'body_pose', 'left_hand_pose',
                           'right_hand_pose', 'jaw_pose', 'transl', 'full_pose'])


class SMAL(nn.Module):

    NUM_JOINTS = 32
    SHAPE_SPACE_DIM = 10

    def __init__(
        self, model_path: str,
        data_struct: Optional[Struct] = None,
        create_betas: bool = True,
        betas: Optional[Tensor] = None,
        num_betas: int = 10,
        create_global_orient: bool = True,
        global_orient: Optional[Tensor] = None,
        create_body_pose: bool = True,
        body_pose: Optional[Tensor] = None,
        create_transl: bool = True,
        transl: Optional[Tensor] = None,
        dtype=torch.float32,
        batch_size: int = 1,
        joint_mapper=None,
        vertex_ids: Dict[str, int] = None,
        v_template: Optional[Union[Tensor, Array]] = None,
        **kwargs
    ) -> None:
        ''' SMPL model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_global_orient: bool, optional
                Flag for creating a member variable for the global orientation
                of the body. (default = True)
            global_orient: torch.tensor, optional, Bx3
                The default value for the global orientation variable.
                (default = None)
            create_body_pose: bool, optional
                Flag for creating a member variable for the pose of the body.
                (default = True)
            body_pose: torch.tensor, optional, Bx(Body Joints * 3)
                The default value for the body pose variable.
                (default = None)
            num_betas: int, optional
                Number of shape components to use
                (default = 10).
            create_betas: bool, optional
                Flag for creating a member variable for the shape space
                (default = True).
            betas: torch.tensor, optional, Bx10
                The default value for the shape member variable.
                (default = None)
            create_transl: bool, optional
                Flag for creating a member variable for the translation
                of the body. (default = True)
            transl: torch.tensor, optional, Bx3
                The default value for the transl variable.
                (default = None)
            dtype: torch.dtype, optional
                The data type for the created variables
            batch_size: int, optional
                The batch size used for creating the member variables
            joint_mapper: object, optional
                An object that re-maps the joints. Useful if one wants to
                re-order the SMPL joints to some other convention (e.g. MSCOCO)
                (default = None)
            gender: str, optional
                Which gender to load
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''
        # NOTE: No PCA on the pose space


        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = 'SMAL.{ext}'.format(ext='pkl')
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
                smpl_path)

            with open(smpl_path, 'rb') as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file,
                                                   encoding='latin1'))

        super(SMAL, self).__init__()
        self.batch_size = batch_size
        shapedirs = data_struct.shapedirs
        if (shapedirs.shape[-1] < self.SHAPE_SPACE_DIM):
            print(f'WARNING: You are using a {self.name()} model, with only'
                  f' {shapedirs.shape[-1]} shape coefficients.\n'
                  f'num_betas={num_betas}, shapedirs.shape={shapedirs.shape}, '
                  f'self.SHAPE_SPACE_DIM={self.SHAPE_SPACE_DIM}')
            num_betas = min(num_betas, shapedirs.shape[-1])
        else:
            num_betas = min(num_betas, self.SHAPE_SPACE_DIM)


        self._num_betas = num_betas
        shapedirs = shapedirs[:, :, :num_betas]
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=dtype))

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS['smal']

        self.dtype = dtype

        self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, use_feet_keypoints=False, use_hands=False, **kwargs)

        self.faces = data_struct.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        if create_betas:
            if betas is None:
                default_betas = torch.zeros(
                    [batch_size, self.num_betas], dtype=dtype)
            else:
                if torch.is_tensor(betas):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas, dtype=dtype)

            self.register_parameter(
                'betas', nn.Parameter(default_betas, requires_grad=True))

        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros(
                    [batch_size, 3], dtype=dtype)
            else:
                if torch.is_tensor(global_orient):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(
                        global_orient, dtype=dtype)

            global_orient = nn.Parameter(default_global_orient,
                                         requires_grad=True)
            self.register_parameter('global_orient', global_orient)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = torch.zeros(
                    [batch_size, self.NUM_JOINTS * 3], dtype=dtype)
            else:
                if torch.is_tensor(body_pose):
                    default_body_pose = body_pose.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose,
                                                     dtype=dtype)
            self.register_parameter(
                'body_pose',
                nn.Parameter(default_body_pose, requires_grad=True))

        if create_transl:
            if transl is None:
                default_transl = torch.zeros([batch_size, 3],
                                             dtype=dtype,
                                             requires_grad=True)
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter(
                'transl', nn.Parameter(default_transl, requires_grad=True))

        if v_template is None:
            v_template = data_struct.v_template
        if not torch.is_tensor(v_template):
            v_template = to_tensor(to_np(v_template), dtype=dtype)
        # The vertices of the template model
        self.register_buffer('v_template', v_template)

        j_regressor = to_tensor(to_np(
            data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        num_pose_basis = data_struct.posedirs.shape[-1]
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        lbs_weights = to_tensor(to_np(data_struct.weights), dtype=dtype)
        self.register_buffer('lbs_weights', lbs_weights)

    @property
    def num_betas(self):
        return self._num_betas

    def create_mean_pose(self, data_struct) -> Tensor:
        pass

    def name(self) -> str:
        return 'SMAL'

    @torch.no_grad()
    def reset_params(self, **params_dict) -> None:
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def get_num_verts(self) -> int:
        return self.v_template.shape[0]

    def get_num_faces(self) -> int:
        return self.faces.shape[0]

    def get_num_joints(self) -> int:
        return self.NUM_JOINTS
    def extra_repr(self) -> str:
        msg = [
            f'Number of joints: {self.J_regressor.shape[0]}',
            f'Betas: {self.num_betas}',
        ]
        return '\n'.join(msg)

    def forward_shape(
        self,
        betas: Optional[Tensor] = None,
    ) -> SMALOutput:
        betas = betas if betas is not None else self.betas
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        return SMALOutput(vertices=v_shaped, betas=betas, v_shaped=v_shaped)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SMALOutput:
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)


        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot)

        # Add extra points to the joints (eg keypoints)?
        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMALOutput(vertices=vertices if return_verts else None,
                            global_orient=global_orient,
                            body_pose=body_pose,
                            joints=joints,
                            betas=betas,
                            full_pose=full_pose if return_full_pose else None)

        return output


class HSMAL(SMAL):
    NUM_JOINTS = 35
    """SMAL with more vertices and faces"""
    # Just need to alter the vertex IDs. The rest is the same
    def __init__(self, model_path: str,
        data_struct: Optional[Struct] = None,
        dtype=torch.float32,
        batch_size: int = 1,
        joint_mapper=None,
        vertex_ids: Dict[str, int] = None,
        v_template: Optional[Union[Tensor, Array]] = None,
        ext: str ='pkl',
        high_res: bool = True,
        **kwargs) -> None:        

        self.high_res = high_res
        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = 'HSMAL{}.{ext}'.format("+" if self.high_res else "", ext='pkl')
                hsmal_path = os.path.join(model_path, model_fn)
            else:
                hsmal_path = model_path
            assert osp.exists(hsmal_path), 'Path {} does not exist!'.format(
                hsmal_path)

            with open(hsmal_path, 'rb') as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file,
                                                   encoding='latin1'))

        if vertex_ids is None:
            if self.high_res:
                vertex_ids = VERTEX_IDS['hsmal+']
            else:
                vertex_ids = VERTEX_IDS['hsmal']

        super(HSMAL, self).__init__(
            model_path=model_path, data_struct=data_struct,
            batch_size=batch_size, dtype=dtype, vertex_ids=vertex_ids,
            ext=ext, **kwargs)       
        
        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, use_hands=False, use_feet_keypoints=False, **kwargs)


class VAREN(HSMAL):
    "NOT SURE ITS CORRECT TO INHERIT FROM HSMAL"
    NUM_JOINTS = 37 # results in 38 joints including 0
    SHAPE_SPACE_DIM = 39
    def __init__(self, model_path: str,
        data_struct: Optional[Struct] = None,
        num_betas: int = 39,
        use_muscle_deformations: bool = True,
        shape_betas_for_muscles: int = 2,
        muscle_betas_size: int = 1,
        dtype=torch.float32,
        batch_size: int = 1,
        joint_mapper=None,
        vertex_ids: Dict[str, int] = None,
        v_template: Optional[Union[Tensor, Array]] = None,
        ext: str ='pkl',
        model_file_name: Optional[str] = None,
        muscle_labels_filename: Optional[str] = "varen_muscle_vertex_labels.npy",
        **kwargs) -> None:

        self.use_muscle_deformations = use_muscle_deformations
        self.shape_betas_for_muscles = shape_betas_for_muscles
        self.muscle_betas_size = muscle_betas_size

        if data_struct is None:
            if osp.isdir(model_path):

                model_fn = '{}.{ext}'.format(
                    "VAREN" if model_file_name is None else model_file_name, ext='pkl')
                varen_path = os.path.join(model_path, model_fn)
            else:
                varen_path = model_path
            assert osp.exists(varen_path), 'Path {} does not exist!'.format(
                varen_path)

            with open(varen_path, 'rb') as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file,
                                                   encoding='latin1'))
                
        if vertex_ids is None:
            vertex_ids = VERTEX_IDS['varen']

        super(VAREN, self).__init__(
            model_path=model_path, data_struct=data_struct,
            batch_size=batch_size, dtype=dtype, vertex_ids=vertex_ids,
            ext=ext, high_res=True, **kwargs)       
        
        self.batch_size = batch_size
        shapedirs = data_struct.shapedirs
        if (shapedirs.shape[-1] < self.SHAPE_SPACE_DIM):
            print(f'WARNING: You are using a {self.name()} model, with only'
                  f' {shapedirs.shape[-1]} shape coefficients.\n'
                  f'num_betas={num_betas}, shapedirs.shape={shapedirs.shape}, '
                  f'self.SHAPE_SPACE_DIM={self.SHAPE_SPACE_DIM}')
            num_betas = min(num_betas, shapedirs.shape[-1])
        else:
            num_betas = min(num_betas, self.SHAPE_SPACE_DIM)


        self._num_betas = num_betas
        shapedirs = shapedirs[:, :, :num_betas]
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=dtype))

        # Add additional information about the part segmentation
        if hasattr(data_struct, 'parts'): 
            self.parts = data_struct.parts
            self.partSet = range(len(self.parts))

        if hasattr(data_struct, 'part2bodyPoints'):
            self.part2bodyPoints = data_struct.part2bodyPoints
        
        if hasattr(data_struct, 'colors_names'):
            self.colors_names = data_struct.colors_names
        
        if hasattr(data_struct, 'seg'):
            self.seg = data_struct.seg

        self.vertex_joint_selector.extra_joints_idxs = to_tensor(
            list(VERTEX_IDS['varen'].values()), dtype=torch.long)
        
        
        if self.use_muscle_deformations:
            muscle_labels_path = osp.join(model_path, muscle_labels_filename)
            # here
            ckpt_path = osp.join(model_path,'varen.pth')
            self.load_VAREN_muscle_model(checkpoint_path=ckpt_path, muscle_labels_path=muscle_labels_path)
            
        # I think thats all for 


    def forward(self, 
                betas: Optional[Tensor] = None,
                body_pose: Optional[Tensor] = None,
                global_orient: Optional[Tensor] = None,
                transl: Optional[Tensor] = None, 
                return_verts: bool = True, 
                return_full_pose: bool = False, 
                pose2rot: bool = True,
                **kwargs):
        
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, 'transl')

        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        # Muscle Predictor forward pass
        
        if self.use_muscle_deformations:
            # A set of decoders, one for each muscle

            betas_muscle, A = self.betas_muscle_predictor.forward(full_pose, betas) # correct.

            muscle_deformer = MuscleDeformer(betas_muscle, self.Bm, self.muscle_idxs)
        else:
            muscle_deformer = None

        vertices, joints, mdv = lbs(betas, full_pose, self.v_template,
                            self.shapedirs, self.posedirs,
                            self.J_regressor, self.parents,
                            self.lbs_weights, pose2rot=pose2rot,
                            muscle_deformer=muscle_deformer)

        # Add extra points to the joints (eg keypoints)?
        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
        
        # Separate joints from surface keypoints
        
        output = VARENOutput(vertices=vertices if return_verts else None,
                        global_orient=global_orient,
                        body_pose=body_pose,
                        joints=joints[:,:self.NUM_JOINTS],
                        surface_keypoints=joints[:,self.NUM_JOINTS:],
                        body_betas=betas,
                        muscle_betas=betas_muscle if self.use_muscle_deformations else None,
                        full_pose=full_pose if return_full_pose else None,
                        muscle_acitivations=A if self.use_muscle_deformations else None,
                        mdv=mdv if self.use_muscle_deformations else None)
        return output

    def define_muscle_deformations_variables(self, muscle_labels_path=None):
        '''
        '''



        self.muscle_labels = np.load(open(muscle_labels_path, 'rb'))

        self.num_muscles = np.max(self.muscle_labels) + 1

        self.muscle_parts = ['LScapula', 'RScapula', 'Spine1', 'Spine2', 'LBLeg1', 'LBLeg2', 'LBLeg3', 'Neck1', 'Neck2', 'Neck', 'Spine', 'LFLeg1', 'LFLeg2', 'LFLeg3', 'RFLeg2', 'RFLeg3', 'RFLeg1', 'Pelvis', 'RBLeg2', 'RBLeg3', 'RBLeg1', 'Head']
        self.muscle_parts_idx = []
        all_idxs = []

        for pa in self.muscle_parts:
            self.muscle_parts_idx += [self.parts[pa]]
            if pa == 'Head':
                all_idxs += list(POLL_VERT_IDS)
            else:
                all_idxs += list(self.part2bodyPoints[self.parts[pa]])
        self.all_muscle_idxs = all_idxs

        # Define the vertices that have no muscle to be associated
        num_joints = self.get_num_joints()

        # Define part-muscle assciation function
        muscle_associations = torch.zeros((num_joints, self.num_muscles))
        
        # Only assign for the parts that we consider affect the muscles (muscle_parts)
        for part in self.muscle_parts_idx:
            # Vertices of this part
            part_vert_ids = self.part2bodyPoints[part]
            labels = np.unique(self.muscle_labels[part_vert_ids])
            
            muscle_associations[part-1, labels] += 1
            
            parent = self.parents[part]

            if parent < num_joints:
                muscle_associations[parent-1,labels] += 1
                
                idx = np.where(self.parents==part)[0]
                for k in idx:
                    muscle_associations[k-1, labels] += 1

        muscle_associations = muscle_associations / torch.max(muscle_associations)

        # probably a more effecient way to do this

        # Define the indices of the vertices that belong to each muscle
        self.muscle_idxs = [None]*self.num_muscles
        for i in range(self.num_muscles):
            self.muscle_idxs[i] = list(set(all_idxs) & set(np.where(self.muscle_labels==i)[0]))

        # What is this?
        self.Bm = torch.nn.ModuleList() 
        for i in range(self.num_muscles):
            pose_d = 4
            self.Bm.append(
                nn.Sequential(
                    nn.Linear(
                        self.muscle_betas_size * num_joints * pose_d + self.shape_betas_for_muscles,
                        len(self.muscle_idxs[i])*3,
                        bias=False
                        )
                    )
                )

            for m in self.Bm[i].modules():
                if isinstance(m, nn.Linear):
                   torch.nn.init.normal_(m.weight, mean=0.0, std=0.001)
                   if m.bias is not None:
                       m.bias.data.zero_()

        return muscle_associations
    

    def load_VAREN_muscle_model(self, checkpoint_path, muscle_labels_path):
        A = self.define_muscle_deformations_variables(muscle_labels_path=muscle_labels_path)
        self.betas_muscle_predictor = BetasMusclePredictor(
            muscle_associations = A,
            shape_beta_for_muscles = self.shape_betas_for_muscles
            )
        print("Loading model from: ", checkpoint_path)
        chkpt = torch.load(checkpoint_path, weights_only=True)
        self.Bm.load_state_dict(chkpt['Bm']) # Should pull what it needs
        self.betas_muscle_predictor.load_state_dict(chkpt['betas_muscle_predictor']) # Should pull what it needs


    @property
    def keypoint_information(self):
        """
        Returns name of Keypoint along with vertex index.
        """
        return VERTEX_IDS['varen']

# Get this part working. Figure out the opts and stuff.
# Probably put the rest of the arguments from opts in the kwargs on VAREN and define them here. Just upack the kwargs on call
class BetasMusclePredictor(nn.Module):
    def __init__(self, muscle_associations, shape_beta_for_muscles, debug=False):
        super(BetasMusclePredictor, self).__init__()
        #self.opts = opts

        self.shape_betas_for_muscles = shape_beta_for_muscles
        self.num_parts, self.num_muscle = muscle_associations.shape
        r_dim = 4 # dimension of rotation -> 4 = quaternion
        num_pose = self.num_parts * 4 # why 4?
        self.num_pose = num_pose

        

        self.muscledef = nn.Linear(self.num_pose + self.shape_betas_for_muscles, self.num_muscle, bias=False)
        torch.nn.init.normal_(self.muscledef.weight, mean=0.0, std=0.001)
        
        A_here = torch.zeros(self.num_muscle, self.num_pose)
        if debug:
            np.save('A_init.npy', muscle_associations.detach().cpu().numpy())

        if self.shape_betas_for_muscles > 0:
            A_here = torch.zeros(
                self.num_muscle,
                self.num_pose + self.shape_betas_for_muscles
                )
            
        for p in range(self.num_parts):
            # What is r dim?
            for k in range(r_dim):
                A_here[:, r_dim * p + k] = muscle_associations[p, :]

        if self.shape_betas_for_muscles > 0:
            A_here[:,self.num_pose:] = 1

        self.A = torch.nn.Parameter(A_here, requires_grad=True) # We learn this. Why do we copy it?

    def forward(self, pose, betas):
        tensor_b = axis_angle_to_quaternion(pose[:,3:].view(-1,self.num_parts,3)).view(-1,self.num_parts*4)
        if self.shape_betas_for_muscles > 0:
            tensor_b = torch.cat((tensor_b, betas[:,:self.shape_betas_for_muscles]),dim=1)

        tensor_a = self.A*self.muscledef.weight

        
        tensor_a = tensor_a.unsqueeze(0)
        tensor_a = tensor_a.expand(pose.shape[0], -1, -1)
        tensor_b = tensor_b.unsqueeze(1)
        tensor_b = tensor_b.expand(-1,self.num_muscle,-1)
        betas_muscle = tensor_a * tensor_b

        return betas_muscle, self.A*self.muscledef.weight