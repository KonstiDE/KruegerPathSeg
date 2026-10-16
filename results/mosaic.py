import os

import matplotlib.pyplot as plt
import rasterio as rio
import shutup
import PIL.Image
import torch

import config.config as cfg
import numpy as np

from tqdm import tqdm

from model.dense121_unet.dense121_unet_model import DenseNet121
from model.unet.unet_model import UNET

from preprocessing.tools import raw_slicing

from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = None
shutup.please()


def generate_result(model, part_path, version, variant):
    tif = rio.open(os.path.join(part_path, "rgb_clipped.tif"))
    tif_array = tif.read().astype(np.uint8)

    if len(tif.indexes) > 3:
        tif_array = tif_array[0:3, :, :]

    mosaic_width_index = (tif_array.shape[1] // cfg.blob_size()) + 1
    mosaic_height_index = (tif_array.shape[2] // cfg.blob_size()) + 1

    arrays = raw_slicing(tif_array, cfg.blob_size())

    result_image = Image.new("F", (mosaic_height_index * cfg.blob_size(), mosaic_width_index * cfg.blob_size()))

    loop = tqdm(arrays)

    cw = 0
    ch = 0
    for data in loop:
        data = torch.Tensor(data).unsqueeze(0).cuda()

        data, degree, distance = model(data)
        data = torch.sigmoid(data)

        data = data.squeeze(0).squeeze(0).detach().cpu().numpy()

        if cw == mosaic_height_index:
            ch += 1
            cw = 0

        result_image.paste(Image.fromarray(data), (cw * cfg.blob_size(), ch * cfg.blob_size()))

        cw += 1

    result_image = result_image.crop((0, 0, tif_array.shape[2], tif_array.shape[1]))

    matrix = np.array(result_image.getdata())
    matrix = matrix.reshape((1, tif_array.shape[1], tif_array.shape[2]))

    # plt.imshow(matrix.squeeze(0))
    # plt.show()

    mosaic = rio.open(
        os.path.join("/home/caipi/Desktop/mosaic_{}_{}.tif".format(variant, version)),
        'w+',
        driver='GTiff',
        width=tif.width,
        height=tif.height,
        count=1,
        dtype=str(rio.float32),
        crs=tif.crs,
        transform=tif.transform
    )

    mosaic.write(matrix)
    mosaic.close()


if __name__ == '__main__':
    immense_config = [
        ["denseunet", False, False, False, "buffer", "checkpoint_000015", "model_epoch16.pt"]
    ]

    for single_config in immense_config:
        variant, aspp, attention, aux, version, checkpoint, epoch_str = single_config

        if variant == "unet":
            model = UNET(in_channels=3, out_channels=1).cuda()
        else:
            model = DenseNet121(in_channels=3, out_channels=1, aspp=aspp, attention=attention, aux=aux).cuda()

        state_dict = torch.load(os.path.join(
            "/media/caipi/data/Master/svalbard/model_epoch31.pt"
        ))["net_state_dict"]
        model.load_state_dict(state_dict)
        model.eval()

        generate_result(
            model=model,
            part_path="/media/caipi/data/Master/svalbard",
            version=version,
            variant=variant
        )
