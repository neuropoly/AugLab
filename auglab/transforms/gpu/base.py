from kornia.augmentation import RandomGamma

from kornia.augmentation._3d.base import RigidAffineAugmentationBase3D
from kornia.core import Tensor
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from kornia.augmentation import AugmentationSequential
from kornia.augmentation.container.ops import MaskSequentialOps
from kornia.augmentation.container.params import ParamItem
import kornia.augmentation as K
from kornia.augmentation.base import _AugmentationBase
from kornia.constants import DataKey
from kornia.core import Module, Tensor
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from typing import Any, Dict, List, Optional

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
    def transform_masks(
        self, input: Tensor, params: List[ParamItem], extra_args: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        for param in params:
            module = self.get_submodule(param.name)
            input = MaskSequentialOpsCustom.transform(input, module=module, param=param, extra_args=extra_args)
        return input

class MaskSequentialOpsCustom(MaskSequentialOps):
    @classmethod
    def transform(
        cls, input: Tensor, module: Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
            extra_args: Optional dictionary of extra arguments with specific options for different input types.
        """
        if extra_args is None:
            extra_args = {}

        if isinstance(module, (K.GeometricAugmentationBase2D,)):
            input = module.transform_masks(
                input,
                params=cls.get_instance_module_param(param),
                flags=module.flags,
                transform=module.transform_matrix,
                **extra_args,
            )

        elif isinstance(module, (K.RigidAffineAugmentationBase3D,)):
            input = module.transform_masks(
                input,
                params=cls.get_instance_module_param(param),
                flags=module.flags,
                transform=module.transform_matrix,
                **extra_args,
            )

        elif isinstance(module, K.RandomTransplantation):
            input = module(input, params=cls.get_instance_module_param(param), data_keys=[DataKey.MASK], **extra_args)

        elif isinstance(module, (_AugmentationBase)):
            input = module.transform_masks(
                input, params=cls.get_instance_module_param(param), flags=module.flags, **extra_args
            )

        elif isinstance(module, K.ImageSequential) and not module.is_intensity_only():
            input = module.transform_masks(input, params=cls.get_sequential_module_param(param), extra_args=extra_args)

        elif isinstance(module, K.container.ImageSequentialBase):
            input = module.transform_masks(input, params=cls.get_sequential_module_param(param), extra_args=extra_args)

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            input = MaskSequentialOps.transform(input, module=module.op, param=param, extra_args=extra_args)

        return input