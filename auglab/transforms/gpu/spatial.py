from kornia.augmentation import RandomAffine3D
from kornia.constants import Resample
from kornia.core import Tensor

from typing import Any, Dict, Optional


class RandomAffine3DCustom(RandomAffine3D):
    def apply_non_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are transformed.

        Note:
            Convert "resample" arguments to "nearest" by default.

        """
        resample_method: Optional[Resample]
        if "resample" in flags:
            resample_method = flags["resample"]
            flags["resample"] = Resample.get("nearest")
        output = self.apply_transform(input, params, flags, transform)
        if resample_method is not None:
            flags["resample"] = resample_method
        return output