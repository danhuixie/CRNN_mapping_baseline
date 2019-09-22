import scipy.io as sio
import os
import progressbar
import torch
root_path = '/data/yangyang/data/SPEECH_ENHANCE_DATA/'
data_path = [root_path + 'tr/', root_path + 'tt/', root_path + 'cv/']
remove_data_num = {root_path + 'tr/': 0,
                   root_path + 'tt/': 0,
                   root_path + 'cv/': 0}
if __name__ == '__main__':
    a = [1, 2, 3]
    b = torch.Tensor(a).cuda("cuda:3")
    print(b)

    # for item in data_path:
    #     data_file = os.listdir(item)
    #     # bar = progressbar.ProgressBar(0, len(data_file))
    #     # bar.start()
    #     tmp = 0
    #     for speech_file in data_file:
    #         # bar.update(tmp)
    #         tmp += 1
    #         try:
    #             data = sio.loadmat(item + speech_file)
    #             speech = data['speech']
    #             noise = data['noise']
    #         except:
    #             os.remove(item + speech_file)
    #             remove_data_num[item] += 1
    #     # bar.finish()
    # file = open('res.txt', mode='w')
    # file.write(root_path + 'tr/: ' + remove_data_num[remove_data_num[root_path + 'tr/']])
    # file.write(root_path + 'tt/: ' + remove_data_num[remove_data_num[root_path + 'tt/']])
    # file.write(root_path + 'cv/: ' + remove_data_num[remove_data_num[root_path + 'cv/']])

