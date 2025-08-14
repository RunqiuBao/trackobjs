from pathlib import Path
import torch

import ops
from .files import increment_path
from PIL import Image


def save_one_box(
    xyxy,
    im,
    file: Path = Path("im.jpg"),
    gain: float = 1.02,
    pad: int = 10,
    square: bool = False,
    BGR: bool = False,
    save: bool = True,
):
    """
    Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop.

    This function takes a bounding box and an image, and then saves a cropped portion of the image according
    to the bounding box. Optionally, the crop can be squared, and the function allows for gain and padding
    adjustments to the bounding box.

    Args:
        xyxy (torch.Tensor | list): A tensor or list representing the bounding box in xyxy format.
        im (np.ndarray): The input image.
        file (Path, optional): The path where the cropped image will be saved.
        gain (float, optional): A multiplicative factor to increase the size of the bounding box.
        pad (int, optional): The number of pixels to add to the width and height of the bounding box.
        square (bool, optional): If True, the bounding box will be transformed into a square.
        BGR (bool, optional): If True, the image will be returned in BGR format, otherwise in RGB.
        save (bool, optional): If True, the cropped image will be saved to disk.

    Returns:
        (np.ndarray): The cropped image.

    Examples:
        >>> from ultralytics.utils.plotting import save_one_box
        >>> xyxy = [50, 50, 150, 150]
        >>> im = cv2.imread("image.jpg")
        >>> cropped_im = save_one_box(xyxy, im, file="cropped.jpg", square=True)
    """
    if not isinstance(xyxy, torch.Tensor):  # may be list
        xyxy = torch.stack(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1, 4))  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = ops.xywh2xyxy(b).long()
    xyxy = ops.clip_boxes(xyxy, im.shape)
    grayscale = im.shape[2] == 1  # grayscale image
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR or grayscale else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix(".jpg"))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        crop = crop.squeeze(-1) if grayscale else crop[..., ::-1] if BGR else crop
        Image.fromarray(crop).save(f, quality=95, subsampling=0)  # save RGB
    return crop

