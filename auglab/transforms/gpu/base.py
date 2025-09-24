import warnings
from kornia.augmentation import RandomGamma

from kornia.augmentation._2d.base import RigidAffineAugmentationBase2D
from kornia.augmentation._3d.base import RigidAffineAugmentationBase3D
from kornia.augmentation._3d.base import AugmentationBase3D, RigidAffineAugmentationBase3D
from kornia.augmentation.base import _AugmentationBase
from kornia.constants import DataKey, Resample
from kornia.core import Tensor
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints
from kornia.augmentation.container.patch import PatchSequential
from kornia.augmentation.container.video import VideoSequential
from kornia.augmentation.container.image import ImageSequential
from kornia.augmentation.container.ops import AugmentationSequentialOps

from kornia.augmentation import AugmentationSequential
from kornia.augmentation.container.ops import MaskSequentialOps
from kornia.augmentation.container.params import ParamItem
import kornia.augmentation as K
from kornia.augmentation.base import _AugmentationBase
from kornia.constants import DataKey
from kornia.core import Module, Tensor
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import copy

DataType = Union[Tensor, List[Tensor], Boxes, Keypoints]
SequenceDataType = Union[List[Tensor], List[List[Tensor]], List[Boxes], List[Keypoints]]

class ImageOnlyTransform(RigidAffineAugmentationBase3D):
    r"""ImageOnlyTransform base class for customized image-only transformations.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.

    """

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return self.identity_matrix(input)

    def apply_non_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # For the images where batch_prob == False.
        return input

    def apply_non_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input

    def apply_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input

    def apply_non_transform_boxes(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes:
        return input

    def apply_transform_boxes(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes:
        return input

    def apply_non_transform_keypoint(
        self, input: Keypoints, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Keypoints:
        return input

    def apply_transform_keypoint(
        self, input: Keypoints, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Keypoints:
        return input

    def apply_non_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input

    def apply_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input

class AugmentationSequentialCustom(AugmentationSequential):
    """Custom AugmentationSequential to handle masks augmentations."""
    def __init__(
        self,
        *args: Union[_AugmentationBase, ImageSequential],
        data_keys: Optional[Union[Sequence[str], Sequence[int], Sequence[DataKey]]] = (DataKey.INPUT,),
        same_on_batch: Optional[bool] = None,
        keepdim: Optional[bool] = None,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
        random_apply_weights: Optional[List[float]] = None,
        transformation_matrix_mode: str = "silent",
        extra_args: Optional[Dict[DataKey, Dict[str, Any]]] = None,
    ) -> None:
        self._transform_matrix: Optional[Tensor]
        self._transform_matrices: List[Optional[Tensor]] = []

        super().__init__(
            *args,
            same_on_batch=same_on_batch,
            keepdim=keepdim,
            random_apply=random_apply,
            random_apply_weights=random_apply_weights,
        )

        self._parse_transformation_matrix_mode(transformation_matrix_mode)

        self._valid_ops_for_transform_computation: Tuple[Any, ...] = (
            RigidAffineAugmentationBase2D,
            RigidAffineAugmentationBase3D,
            AugmentationSequential,
        )

        self.data_keys: Optional[List[DataKey]]
        if data_keys is not None:
            self.data_keys = [DataKey.get(inp) for inp in data_keys]
        else:
            self.data_keys = data_keys

        if self.data_keys:
            if any(in_type not in DataKey for in_type in self.data_keys):
                raise AssertionError(f"`data_keys` must be in {DataKey}. Got {self.data_keys}.")

            if self.data_keys[0] != DataKey.INPUT:
                raise NotImplementedError(f"The first input must be {DataKey.INPUT}.")

        self.transform_op = AugmentationSequentialOpsCustom(self.data_keys)

        self.contains_video_sequential: bool = False
        self.contains_3d_augmentation: bool = False
        for arg in args:
            if isinstance(arg, PatchSequential) and not arg.is_intensity_only():
                warnings.warn(
                    "Geometric transformation detected in PatchSeqeuntial, which would break bbox, mask.", stacklevel=1
                )
            if isinstance(arg, VideoSequential):
                self.contains_video_sequential = True
            # NOTE: only for images are supported for 3D.
            if isinstance(arg, AugmentationBase3D):
                self.contains_3d_augmentation = True
        self._transform_matrix = None
        self.extra_args = extra_args or {DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}}
    def transform_masks(
        self, input: Tensor, params: List[ParamItem], extra_args: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        for param in params:
            module = self.get_submodule(param.name)
            input = MaskSequentialOpsCustom.transform(input, module=module, param=param, extra_args=extra_args)
        return input

class MaskSequentialOpsCustom(MaskSequentialOps):
    @classmethod
    def transform_list(
        cls, input: List[Tensor], module: Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None
    ) -> List[Tensor]:
        """Apply a transformation with respect to the parameters.

        Args:
            input: list of input tensors.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
            extra_args: Optional dictionary of extra arguments with specific options for different input types.
        """
        if extra_args is None:
            extra_args = {}
        if isinstance(module, (K.GeometricAugmentationBase2D,)):
            tfm_input = []
            params = cls.get_instance_module_param(param)
            params_i = copy.deepcopy(params)
            for i, inp in enumerate(input):
                params_i["batch_prob"] = params["batch_prob"][i]
                tfm_inp = module.transform_masks(
                    inp, params=params_i, flags=module.flags, transform=module.transform_matrix, **extra_args
                )
                tfm_input.append(tfm_inp)
            input = tfm_input

        elif isinstance(module, (K.RigidAffineAugmentationBase3D,)):
            tfm_input = []
            params = cls.get_instance_module_param(param)
            params_i = copy.deepcopy(params)
            for i, inp in enumerate(input):
                params_i["batch_prob"] = params["batch_prob"][i]
                tfm_inp = module.transform_masks(
                    inp, params=params_i, flags=module.flags, transform=module.transform_matrix, **extra_args
                )
                tfm_input.append(tfm_inp)
            input = tfm_input

        elif isinstance(module, (_AugmentationBase)):
            tfm_input = []
            params = cls.get_instance_module_param(param)
            params_i = copy.deepcopy(params)
            for i, inp in enumerate(input):
                params_i["batch_prob"] = params["batch_prob"][i]
                tfm_inp = module.transform_masks(inp, params=params_i, flags=module.flags, **extra_args)
                tfm_input.append(tfm_inp)
            input = tfm_input

        elif isinstance(module, K.ImageSequential) and not module.is_intensity_only():
            tfm_input = []
            seq_params = cls.get_sequential_module_param(param)
            for inp in input:
                tfm_inp = module.transform_masks(inp, params=seq_params, extra_args=extra_args)
                tfm_input.append(tfm_inp)
            input = tfm_input

        elif isinstance(module, K.container.ImageSequentialBase):
            tfm_input = []
            seq_params = cls.get_sequential_module_param(param)
            for inp in input:
                tfm_inp = module.transform_masks(inp, params=seq_params, extra_args=extra_args)
                tfm_input.append(tfm_inp)
            input = tfm_input

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            raise NotImplementedError(
                "The support for list of masks under auto operations are not yet supported. You are welcome to file a"
                " PR in our repo."
            )
        return input

class AugmentationSequentialOpsCustom(AugmentationSequentialOps):
    def transform(
        self,
        *arg: DataType,
        module: Module,
        param: ParamItem,
        extra_args: Dict[DataKey, Dict[str, Any]],
        data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None,
    ) -> Union[DataType, SequenceDataType]:
        _data_keys = self.preproc_datakeys(data_keys)

        if isinstance(module, K.RandomTransplantation):
            # For transforms which require the full input to calculate the parameters (e.g. RandomTransplantation)
            param = ParamItem(
                name=param.name,
                data=module.params_from_input(
                    *arg,  # type: ignore[arg-type]
                    data_keys=_data_keys,
                    params=param.data,  # type: ignore[arg-type]
                    extra_args=extra_args,
                ),
            )

        outputs = []
        for inp, dcate in zip(arg, _data_keys):
            op = self._get_op(dcate)
            extra_arg = extra_args.get(dcate, {})
            if dcate.name == "MASK" and isinstance(inp, list):
                outputs.append(MaskSequentialOpsCustom.transform_list(inp, module, param=param, extra_args=extra_arg))
            else:
                outputs.append(op.transform(inp, module, param=param, extra_args=extra_arg))
        if len(outputs) == 1 and isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs