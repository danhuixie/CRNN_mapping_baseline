import sys
import os

import progressbar
import torch
from scipy.io import loadmat
import soundfile as sf
from net.module import CRNN

from utils.stft_istft import STFT
from utils.util import get_alpha

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from config import *
import os
from subprocess import PIPE, Popen

def validation(path, net):
    net.eval()
    files = os.listdir(path)
    pesq_unprocess = 0
    pesq_res = 0
    bar = progressbar.ProgressBar(0, len(files))
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH).cuda(CUDA_ID[0])
    for i in range(len(files)):
        bar.update(i)
        with torch.no_grad():
            speech = loadmat(path + files[i])['speech']
            noise = loadmat(path + files[i])['noise']
            mix = speech + noise

            sf.write(files[i][:-4] + '_clean.wav', speech, 16000)
            sf.write(files[i][:-4] + '_mix.wav', mix, 16000)

            c = get_alpha(mix)
            mix *= c
            speech *= c
            noise *= c

            speech = stft.transform(torch.Tensor(speech.T).cuda(CUDA_ID[0]))
            mix = stft.transform(torch.Tensor(mix.T).cuda(CUDA_ID[0]))
            noise = stft.transform(torch.Tensor(noise.T).cuda(CUDA_ID[0]))

            mix_real = mix[:, :, :, 0]
            mix_imag = mix[:, :, :, 1]
            mix_mag = torch.sqrt(mix_real ** 2 + mix_imag ** 2)


            # mix_(T,F)
            mix_mag = mix_mag.unsqueeze(0).cuda(CUDA_ID[0])
            # output(1, T, F)

            mapping_out = net(mix_mag)

            res_real = mapping_out * mix_real / mix_mag.squeeze(0)
            res_imag = mapping_out * mix_imag / mix_mag.squeeze(0)

            res = torch.stack([res_real, res_imag], 3)
            output = stft.inverse(res)

            output = output.permute(1, 0).detach().cpu().numpy()

            # 写入的必须是（F,T）istft之后的
            sf.write(files[i][:-4] + '_est.wav', output / c, 16000)
    bar.finish()
    net.train()
    return


if __name__ == '__main__':
    net = CRNN()
    net = torch.load(MODEL_STORE + 'model_43000.pkl')
    net.eval()
    validation(VALIDATION_DATA_PATH, net)
    commend_line = MATLAB_HOME + ' -nodesktop -nosplash -r \'try comput_metrics; catch; end; quit \''
    process = Popen(args=commend_line, stdout=PIPE, shell=True)



