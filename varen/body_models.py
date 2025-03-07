from typing import Optional, Dict, Union
import os
import os.path as osp

import pickle

import numpy as np

import torch
import torch.nn as nn

from .lbs import lbs, blend_shapes

from .vertex_ids import vertex_ids as VERTEX_IDS


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
                re-order the model joints to some other convention (e.g. MSCOCO)
                (default = None)
            gender: str, optional
                Which gender to load
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''
        # NOTE: No PCA on the pose space

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                    [batch_size, self.num_betas], dtype=dtype, device=device)
            else:
                if torch.is_tensor(betas):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas, dtype=dtype, device=device)

            self.register_parameter(
                'betas', nn.Parameter(default_betas, requires_grad=True))

        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros(
                    [batch_size, 3], dtype=dtype, device=device)
            else:
                if torch.is_tensor(global_orient):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(
                        global_orient, dtype=dtype, device=device)

            global_orient = nn.Parameter(default_global_orient,
                                         requires_grad=True)
            self.register_parameter('global_orient', global_orient)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = torch.zeros(
                    [batch_size, self.NUM_JOINTS * 3], dtype=dtype, device=device)
            else:
                if torch.is_tensor(body_pose):
                    default_body_pose = body_pose.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose,
                                                     dtype=dtype, device=device)
            self.register_parameter(
                'body_pose',
                nn.Parameter(default_body_pose, requires_grad=True))

        if create_transl:
            if transl is None:
                default_transl = torch.zeros([batch_size, 3],
                                             dtype=dtype,
                                             requires_grad=True, device=device)
            else:
                default_transl = torch.tensor(transl, dtype=dtype, device=device)
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
        ''' Forward pass for the SMAL model

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
            batch_size=batch_size, v_template=v_template, joint_mapper=joint_mapper,
            dtype=dtype, vertex_ids=vertex_ids,
            ext=ext, **kwargs)       
        
        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, use_hands=False, use_feet_keypoints=False, **kwargs)


class VAREN(HSMAL):
    """
    VAREN (Very Accurate and Realistic Equine Network) class that extends the HSMAL class.
    
    This model is designed for equine (horse) body shape and pose estimation. It utilizes a similar approach to SMPL,
    but is specifically tailored for horses. The model includes the ability to simulate muscle deformations using a neural 
    network, making it suitable for accurate and realistic representations of equine structures and movements.
    
    Args:
        model_path (str): Path to the model directory or file.
        data_struct (Optional[Struct], optional): The data structure containing model information (default is None).
        num_betas (int, optional): The number of shape coefficients to use for the model (default is 39).
        use_muscle_deformations (bool, optional): Whether to include muscle deformations (default is True).
        shape_betas_for_muscles (int, optional): Number of shape betas for muscle modeling (default is 2).
        muscle_betas_size (int, optional): The size of muscle betas (default is 1).
        dtype (torch.dtype, optional): The data type of tensors (default is torch.float32).
        batch_size (int, optional): The batch size for the model (default is 1).
        joint_mapper (Optional[callable], optional): A function to map joints (default is None).
        vertex_ids (Optional[Dict[str, int]], optional): A dictionary of vertex IDs for different parts of the horse's body (default is None).
        v_template (Optional[Union[Tensor, Array]], optional): Template vertex data for the model (default is None).
        ext (str, optional): File extension for the model data (default is 'pkl').
        model_file_name (Optional[str], optional): The file name for the model (default is None).
        ckpt_file (Optional[str], optional): Path to a checkpoint file for loading the model (default is None).
        **kwargs: Additional arguments passed to the parent class initialization.
    """

    NUM_JOINTS = 37 # Results in 38 joints including 0 (for equine body structure)
    SHAPE_SPACE_DIM = 39 # The dimensionality of the shape space for the model
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
                 ckpt_file: Optional[str] = 'varen.pth',
                 **kwargs) -> None:
        """
        Initializes the VAREN model, which is a highly accurate and realistic model for equine body shape and pose 
        estimation with optional muscle deformation simulation.
        
        Args:
            model_path (str): The path to the directory containing the model data or the model file.
            data_struct (Optional[Struct], optional): Data structure containing model details (default is None).
            num_betas (int, optional): Number of shape coefficients for the model (default is 39).
            use_muscle_deformations (bool, optional): Whether to use muscle deformations (default is True).
            shape_betas_for_muscles (int, optional): Number of shape betas for muscle modeling (default is 2).
            muscle_betas_size (int, optional): The size of muscle betas (default is 1).
            dtype (torch.dtype, optional): Data type for the model tensors (default is torch.float32).
            batch_size (int, optional): Batch size (default is 1).
            joint_mapper (Optional[callable], optional): A mapping function for joints (default is None).
            vertex_ids (Optional[Dict[str, int]], optional): Dictionary mapping body parts to vertex IDs (default is None).
            v_template (Optional[Union[Tensor, Array]], optional): Template vertices (default is None).
            ext (str, optional): File extension for the model (default is 'pkl').
            model file_name (Optional[str], optional): The specific model filename (default is None).
            ckpt_file (Optional[str], optional): Path to the checkpoint file for model loading (default is None).
            **kwargs: Additional arguments passed to the parent class HSMAL initialization.
        """

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

            with open(varen_path, 'rb') as file:
                data_struct = Struct(**pickle.load(file,
                                                   encoding='latin1'))
                
        if vertex_ids is None:
            vertex_ids = VERTEX_IDS['varen']

        super(VAREN, self).__init__(
            model_path=model_path, data_struct=data_struct,
            batch_size=batch_size, v_template=v_template, joint_mapper=joint_mapper, 
            dtype=dtype, vertex_ids=vertex_ids,
            ext=ext, high_res=True, num_betas = num_betas, **kwargs)       
        
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

        # Likely depricated for the below
        if hasattr(data_struct, 'part2vertices'):
            self.part2vertices = data_struct.part2vertices
        
        if hasattr(data_struct, 'colors_names'):
            self.colors_names = data_struct.colors_names
        
        if hasattr(data_struct, 'seg'):
            self.seg = data_struct.seg

        if hasattr(data_struct, 'muscle_labels'):
            self.muscle_labels = data_struct.muscle_labels

        self.vertex_joint_selector.extra_joints_idxs = to_tensor(
            list(VERTEX_IDS['varen'].values()), dtype=torch.long)
        
        
        if self.use_muscle_deformations:
            # Neural muscle deformer added to self
            self.create_neural_muscle_deformer()
            
            # If path exists, load checkpoint
            if ckpt_file is not None:
                ckpt_path = osp.join(model_path, ckpt_file)
            else:
                ckpt_path = None

            if ckpt_path is not None:
                print("Loading VAREN Muscle Model from: ", ckpt_path)
                chkpt = torch.load(ckpt_path, weights_only=True)
                self.Bm.load_state_dict(chkpt['Bm']) # Load Bm weights (Muscle Betas) -> Deforms vertices based on betas
                self.betas_muscle_predictor.load_state_dict(chkpt['betas_muscle_predictor']) # Load betas muscle predictor weights
            

    def create_neural_muscle_deformer(self) -> None:
        """
        Creates and initializes the neural muscle deformation model using muscle labels and associations.
        
        This method defines the muscle deformations based on the muscle labels and shape betas for muscles.
        
        Args:
            model_path (str): The path to the model directory.
            muscle_labels_filename (str): The filename of the muscle vertex labels.
        """

        muscle_associations = self.define_muscle_deformations_variables()

        # Predict Muscle Betas based on pose and shape
        self.betas_muscle_predictor = MuscleBetaPredictor(
            muscle_associations = muscle_associations,
            shape_beta_for_muscles = self.shape_betas_for_muscles
        )
        
        # Predict offsets based on muscle betas
        self.Bm = self.create_Bm()

    def forward(self, 
                betas: Optional[Tensor] = None,
                body_pose: Optional[Tensor] = None,
                global_orient: Optional[Tensor] = None,
                transl: Optional[Tensor] = None, 
                return_verts: bool = True, 
                return_full_pose: bool = False, 
                pose2rot: bool = True,
                **kwargs) -> VARENOutput:
        """
        Forward pass for the VAREN model. Computes the body shape, pose, and optional muscle deformations for the horse model.

        Args:
            betas (Optional[Tensor], optional): Shape coefficients for the body model (default is None).
            body_pose (Optional[Tensor], optional): Body pose parameters (default is None).
            global_orient (Optional[Tensor], optional): Global orientation parameters (default is None).
            transl (Optional[Tensor], optional): Translation parameters (default is None).
            return_verts (bool, optional): Whether to return the vertices of the model (default is True).
            return_full_pose (bool, optional): Whether to return the full pose including global orientation (default is False).
            pose2rot (bool, optional): Whether to convert pose parameters to rotation matrices (default is True).
            **kwargs: Additional arguments passed to the method.

        Returns:
            VARENOutput: The output of the forward pass, including the computed body shape, pose, vertices, muscle activations, etc.
        """
        
        global_orient = self.global_orient if global_orient is None else global_orient
        body_pose = self.body_pose if body_pose is None else body_pose
        betas = self.betas if betas is None else betas
        transl = self.transl if transl is None else transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)


        # Muscle Predictor forward pass
        muscle_deformer = self.compute_muscle_deformations(full_pose, betas) if self.use_muscle_deformations else None
    
        vertices, joints, mdv = lbs(betas, full_pose, self.v_template,
                            self.shapedirs, self.posedirs,
                            self.J_regressor, self.parents,
                            self.lbs_weights, pose2rot=pose2rot,
                            muscle_deformer=muscle_deformer)

        joints = self.vertex_joint_selector(vertices, joints)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
                
        output = VARENOutput(vertices=vertices if return_verts else None,
                        global_orient=global_orient,
                        body_pose=body_pose,
                        joints=joints[:,:self.NUM_JOINTS],
                        surface_keypoints=joints[:,self.NUM_JOINTS:],
                        body_betas=betas,
                        muscle_betas=muscle_deformer.betas_muscle if self.use_muscle_deformations else None,
                        full_pose=full_pose if return_full_pose else None,
                        muscle_activations=self.A if self.use_muscle_deformations else None,
                        mdv=mdv if self.use_muscle_deformations else None)
        return output
    
    def compute_muscle_deformations(self, full_pose, betas):
        """Computes muscle deformations if enabled."""
        muscle_betas, self.A = self.betas_muscle_predictor.forward(full_pose, betas)
        # Muscle Deformer is a just a Dataclass containing the outputs of MuscleBetasPredictor
        return MuscleDeformer(muscle_betas, self.Bm, self.muscle_vertex_map) 

    def define_muscle_deformations_variables(self) -> torch.Tensor:
        """
        Defines muscle deformation variables, including muscle associations and muscle vertex indices.

        This function:
        - Loads muscle labels and determines the number of distinct muscles.
        - Establishes muscle-to-joint associations for deformation computation.
        - Maps mesh vertices to their corresponding muscles.

        Args:
            muscle_labels_path (Optional[str], optional): Path to the muscle labels file. Defaults to None.

        Returns:
            torch.Tensor: A tensor representing muscle associations for the horse model.
        """

        # Define the anatomical parts associated with muscles
        muscle_parts = [
            'LScapula', 'RScapula', 'Spine1', 'Spine2', 'LBLeg1', 'LBLeg2', 'LBLeg3',
            'Neck1', 'Neck2', 'Neck', 'Spine', 'LFLeg1', 'LFLeg2', 'LFLeg3', 'RFLeg2',
            'RFLeg3', 'RFLeg1', 'Pelvis', 'RBLeg2', 'RBLeg3', 'RBLeg1', 'Head'
        ]

        # Aggregate all muscle-related vertices
        all_muscle_idxs = np.concatenate([self.part2vertices[part] for part in muscle_parts])
        
        self.num_muscles = np.max(self.muscle_labels) + 1
        num_joints = self.get_num_joints()

        # Initialize muscle-to-joint association tensor
        muscle_associations = torch.zeros((num_joints, self.num_muscles))

        # Assign muscle influences to joints based on muscle labels
        for part in muscle_parts:
            part_idx = self.parts[part]  # Get joint index for the part
            part_vertices = self.part2vertices[part]  # Retrieve associated vertices
            labels = np.unique(self.muscle_labels[part_vertices])  # Find unique muscle labels

            # Assign muscles to the corresponding joint
            muscle_associations[part_idx - 1, labels] += 1

            # Also propagate to parent joints
            parent_joint = self.parents[part_idx]
            if parent_joint > -1:
                muscle_associations[parent_joint - 1, labels] += 1

                # Propagate associations to child joints
                for child_idx in np.where(self.parents == part_idx)[0]:
                    muscle_associations[child_idx - 1, labels] += 1

        # Normalize muscle associations for numerical stability
        muscle_associations /= torch.max(muscle_associations)

        # Define the vertex indices associated with each muscle
        muscle_idx_set = set(all_muscle_idxs)
        self.muscle_vertex_map = [
            list(muscle_idx_set & set(np.where(self.muscle_labels == i)[0]))
            for i in range(self.num_muscles)
        ]

        return muscle_associations

    def create_Bm(self):
        """
        Creates and returns a `torch.nn.ModuleList` containing linear layers (muscle deformation networks), 
        one for each muscle.

        Each linear layer maps a combination of muscle pose and shape parameters 
        into per-vertex deformations for the corresponding muscle.

        Logic:
        - Each muscle gets a corresponding layer, unless the muscle maps to zero vertices.
        - Input dimension is based on pose and shape parameters.
        - Output dimension is 3 * number of vertices associated with the muscle (x, y, z per vertex).
        - If a muscle does not affect any vertices (out_dim == 0), a placeholder `None` is stored 
        at that index in the list instead of a real layer (to avoid unnecessary computation).

        Returns:
            torch.nn.ModuleList: A list where each entry is either a neural network layer (for 
            muscles that have associated vertices) or `None` (for muscles that don't).

        Notes:
            - This design choice allows muscles to optionally have no influence (e.g., if they 
            correspond to no vertices in the mesh). This is safer than adding a "no-op" layer 
            since it avoids calling unnecessary zero-output operations.
            - The weight of each linear layer is initialized to a normal distribution with mean 0 
            and standard deviation 0.001.
        """
        num_joints = self.get_num_joints()
        Bm = torch.nn.ModuleList() 
        # Append a layer for each muscle
        for i in range(self.num_muscles):
            pose_d = 4
            input_dim = self.muscle_betas_size * num_joints * pose_d + self.shape_betas_for_muscles
            out_dim = len(self.muscle_vertex_map[i])*3

            # Avoid No Op layers if the muscle relates to no vertices
            if input_dim > 0 and out_dim > 0:
                layer = nn.Linear(input_dim,out_dim,bias=False)
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.001)
                Bm.append(nn.Sequential(layer))
            else:
                Bm.append(None)

        return Bm

    @property
    def keypoint_information(self) -> Dict[str, int]:
        """
        Returns a dictionary containing the name of each keypoint along with the corresponding vertex index.
        
        This includes keypoints for various parts of the horse's body.

        Returns:
            Dict[str, int]: Dictionary mapping keypoint names to their corresponding vertex indices.
        """
        return VERTEX_IDS['varen']



class MuscleBetaPredictor(nn.Module):
    """
    Predicts betas of Muscles based on pose and shape of horse.
    """
    def __init__(self, muscle_associations, shape_beta_for_muscles, dtype=torch.float32):
        super(MuscleBetaPredictor, self).__init__()
    
        self.shape_betas_for_muscles = shape_beta_for_muscles
        self.num_parts, self.num_muscle = muscle_associations.shape
        rot_form_dim = 4 # dimension of rotation -> 4 = quaternion
        self.num_pose = self.num_parts * rot_form_dim
        
        self.muscledef = nn.Linear(self.num_pose + self.shape_betas_for_muscles, self.num_muscle, bias=False)
        torch.nn.init.normal_(self.muscledef.weight, mean=0.0, std=0.001).to(dtype)
        A_here = torch.zeros(self.num_muscle, self.num_pose).to(dtype)
        

        if self.shape_betas_for_muscles > 0:
            A_here = torch.zeros(
                self.num_muscle,
                self.num_pose + self.shape_betas_for_muscles
                )
            
        for p in range(self.num_parts):
            for k in range(rot_form_dim):
                A_here[:, rot_form_dim * p + k] = muscle_associations[p, :]

        if self.shape_betas_for_muscles > 0:
            A_here[:,self.num_pose:] = 1

        self.A = torch.nn.Parameter(A_here, requires_grad=True).to(dtype) # We learn this. Why do we copy it?

    def forward(self, pose, betas):
        pose = pose[:,3:].view(-1,self.num_parts,3)
        tensor_b = axis_angle_to_quaternion(pose).view(-1,self.num_parts*4)
        if self.shape_betas_for_muscles > 0:
            tensor_b = torch.cat((tensor_b, betas[:,:self.shape_betas_for_muscles]),dim=1)

        tensor_a = self.A*self.muscledef.weight

        
        tensor_a = tensor_a.unsqueeze(0)
        tensor_a = tensor_a.expand(pose.shape[0], -1, -1)
        tensor_b = tensor_b.unsqueeze(1)
        tensor_b = tensor_b.expand(-1,self.num_muscle,-1)
        betas_muscle = tensor_a * tensor_b

        return betas_muscle, self.A*self.muscledef.weight
    

