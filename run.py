import cv2
import torch
import numpy as np
import math
from networks import EDSC
import getopt
import sys

assert (int(str('').join(torch.__version__.split('.')[0:3])) >= 100)  # requires at least pytorch version 1.0.0

torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

arguments_strModel = "EDSC_s"
arguments_strModelStateDict = './EDSC_s_l1.ckpt'

arguments_strFirst = './frame10.png'
arguments_strSecond = './frame11.png'
arguments_strOut = './out.png'
arguments_intDevice = 2
arguments_floatTime = 0.1

for strOption, strArgument in \
getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--device' and strArgument != '': arguments_intDevice = int(strArgument)  # device number
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument  # model type
    if strOption == '--model_state' and strArgument != '': arguments_strModelStateDict = strArgument  # path to the model state
    if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument  # path to the first frame
    if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument  # path to the second frame
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument  # path to where the output should be stored
    if strOption == '--time' and strArgument != '': arguments_floatTime = float(strArgument)  # the intermediate time of the synthesized frame

torch.cuda.set_device(arguments_intDevice)


def evaluate(im1_path, im2_path, save_path):
    if arguments_strModel == "EDSC_s":
        GenerateModule = EDSC.Network(isMultiple=False).cuda()
        GenerateModule.load_state_dict(
            torch.load(arguments_strModelStateDict, map_location=lambda storage, loc: storage)['model_state'])
        GenerateModule.eval()

    elif arguments_strModel == "EDSC_m":
        GenerateModule = EDSC.Network(isMultiple=True).cuda()
        GenerateModule.load_state_dict(
            torch.load(arguments_strModelStateDict, map_location=lambda storage, loc: storage)['model_state'])
        GenerateModule.eval()

    with torch.no_grad():
        path1 = im1_path
        path2 = im2_path

        write_path = save_path
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        assert img1.shape == img2.shape

        temp_input_images1 = np.zeros((1, img1.shape[0], img1.shape[1], img1.shape[2]), np.float32)
        temp_input_images2 = np.zeros((1, img1.shape[0], img1.shape[1], img1.shape[2]), np.float32)

        temp_input_images1[0, :, :, :] = img1[:, :, :].astype(np.float32) / 255.0
        temp_input_images2[0, :, :, :] = img2[:, :, :].astype(np.float32) / 255.0

        temp_input_images1 = np.rollaxis(temp_input_images1, 3, 1)
        temp_input_images2 = np.rollaxis(temp_input_images2, 3, 1)

        img1_V = torch.from_numpy(temp_input_images1).cuda()
        img2_V = torch.from_numpy(temp_input_images2).cuda()

        modulePaddingInput = torch.nn.ReplicationPad2d(
            [0, int((math.ceil(img1_V.size(3) / 32.0) * 32 - img1_V.size(3))), 0,
             int((math.ceil(img1_V.size(2) / 32.0) * 32 - img1_V.size(2)))])
        modulePaddingOutput = torch.nn.ReplicationPad2d(
            [0, 0 - int((math.ceil(img1_V.size(3) / 32.0) * 32 - img1_V.size(3))), 0,
             0 - int((math.ceil(img1_V.size(2) / 32.0) * 32 - img1_V.size(2)))])

        img1_V_padded = modulePaddingInput(img1_V)
        img2_V_padded = modulePaddingInput(img2_V)

        if arguments_strModel == 'EDSC_s':
            variableOutput = GenerateModule([img1_V_padded, img2_V_padded])
            variableOutput = modulePaddingOutput(variableOutput)
        elif arguments_strModel == 'EDSC_m':
            time_torch = torch.ones((1, 1, int(img1_V_padded.shape[2] / 2), int(img1_V_padded.shape[3] / 2))) * arguments_floatTime
            variableOutput = GenerateModule([img1_V_padded, img2_V_padded, time_torch.cuda()])
            variableOutput = modulePaddingOutput(variableOutput)

        output = variableOutput.data.permute(0, 2, 3, 1)
        out = output.cpu().clamp(0.0, 1.0).numpy() * 255.0
        result = out.squeeze().astype(np.uint8)
        cv2.imwrite(write_path, result)

    return result


if __name__ == '__main__':
    evaluate(arguments_strFirst, arguments_strSecond, arguments_strOut)
