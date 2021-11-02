import os
import cv2
import time
import glob
import random
import torch
import argparse
import logging
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from datetime import datetime, timedelta
from utils import make_image

from modules.encoder128 import Backbone128
from modules.iib import IIB
from modules.aii_generator import AII1024
from modules.decoder1024 import UnetDecoder1024
from preprocess.mtcnn import MTCNN

mtcnn = MTCNN()


class FaceAlign(TensorDataset):
    def __init__(self, data_path_list, tar_random=True, src_path_list=None, crop_size=(1024, 1024)):
        super(FaceAlign, self).__init__()
        datasets = []
        self.N = []
        self.T = []
        self.tar_random = tar_random
        self.crop_size = crop_size

        for data_path in data_path_list:
            if os.path.exists(data_path):
                image_list = glob.glob(f'{data_path}/*.*g')
                datasets.append(image_list)
                self.N.append(len(image_list))
        self.datasets = datasets

        self.src_datasets = None
        if src_path_list is not None:
            self.src_datasets = []
            for src_path in src_path_list:
                if os.path.exists(src_path):
                    src_list = glob.glob(f'{src_path}/*.*g')
                    self.src_datasets.append(src_list)
                    self.T.append(len(src_list))

        self.transforms = transforms.Compose([
            transforms.Resize(crop_size, interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        crop_size = self.crop_size
        # print(item)
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1

        if self.src_datasets is None:
            image_path_s = self.datasets[idx][item]
            while not (image_path_s.endswith('.png') or image_path_s.endswith('.jpg')):
                image_path_s = random.choice(self.datasets[random.randint(0, len(self.datasets) - 1)])
            Xs = cv2.imread(image_path_s)
            Xs = Image.fromarray(Xs)
            faces = mtcnn.align_multi(Xs, min_face_size=64, crop_size=crop_size)
            if faces is not None:
                Xs = faces[0]
            else:
                Xs = None
                while Xs is None:
                    image_path_s = random.choice(self.datasets[random.randint(0, len(self.datasets) - 1)])
                    Xs = cv2.imread(image_path_s)
                    Xs = Image.fromarray(Xs)
                    faces = mtcnn.align_multi(Xs, min_face_size=64, crop_size=crop_size)
                    if faces is not None:
                        Xs = faces[0]
                    else:
                        Xs = None
        else:  # Another src path
            Xs = None
            while Xs is None:
                image_path_s = random.choice(self.src_datasets[random.randint(0, len(self.src_datasets) - 1)])
                Xs = cv2.imread(image_path_s)
                Xs = Image.fromarray(Xs)
                faces = mtcnn.align_multi(Xs, min_face_size=20., thresholds=[0.1, 0.6, 0.6], crop_size=crop_size)
                if faces is not None:
                    Xs = faces[0]
                else:
                    print('s')
                    Xs = None

        if self.tar_random:
            Xt = None
            while Xt is None:
                image_path_t = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
                Xt = cv2.imread(image_path_t)
                Xt = Image.fromarray(Xt)
                faces = mtcnn.align_multi(Xt, min_face_size=20., thresholds=[0.1, 0.6, 0.6], crop_size=crop_size)
                if faces is not None:
                    Xt = faces[0]
                else:
                    print('t')
                    Xt = None
        else:
            self.datasets_tar = self.datasets[::-1]
            image_path_t = self.datasets_tar[idx][item]
            while not (image_path_t.endswith('.png') or image_path_t.endswith('.jpg')):
                image_path_t = random.choice(self.datasets_tar[random.randint(0, len(self.datasets_tar) - 1)])
            Xt = cv2.imread(image_path_t)
            Xt = Image.fromarray(Xt)
            faces = mtcnn.align_multi(Xt, min_face_size=20., thresholds=[0.1, 0.6, 0.6], crop_size=crop_size)
            if faces is not None:
                Xt = faces[0]
            else:
                Xt = None
                while Xt is None:
                    image_path_t = random.choice(self.datasets_tar[random.randint(0, len(self.datasets_tar) - 1)])
                    Xt = cv2.imread(image_path_t)
                    Xt = Image.fromarray(Xt)
                    faces = mtcnn.align_multi(Xt, min_face_size=20., thresholds=[0.1, 0.6, 0.6], crop_size=crop_size)
                    if faces is not None:
                        Xt = faces[0]
                    else:
                        print('t')
                        Xt = None

        return self.transforms(Xs), self.transforms(Xt), [
            image_path_s.split('/')[-1].split('.')[0], image_path_t.split('/')[-1].split('.')[0]]  #, Xs_id, Xt_id

    def __len__(self):
        return sum(self.N)


def inference_1024(save_dir):
    # load pre-calculated mean and std
    param_dict = []
    for i in range(N + 1):
        state = torch.load(f'.modules/weights128/readout_layer{i}.pth', map_location=device)
        n_samples = state['n_samples'].float()
        std = torch.sqrt(state['s'] / (n_samples - 1)).to(device)

        neuron_nonzero = state['neuron_nonzero'].float()
        active_neurons = (neuron_nonzero / n_samples) > threshold

        param_dict.append([state['m'].to(device), std, active_neurons])
        print(f'readout_layer{i} | ', end='')
    print()

    for idx, (Xs, Xt, Xs_Xt_names) in enumerate(dataloader):
        img_name = '%05d.png' % idx if B != 1 else f'{Xs_Xt_names[0][0]}_{Xs_Xt_names[1][0]}.png'
        print(Xs.shape, Xt.shape)

        start_time = time.time()
        Xs = Xs.to(device)
        Xt = Xt.to(device)

        """# 1. Get Inter-features After One Feed-Forward: batch size is 2 * B, [:B] for Xs and [B:] for Xt"""
        with torch.no_grad():
            X_id = model(
                F.interpolate(torch.cat((Xs, Xt), dim=0)[:, :, 74:950, 74:950], size=[128, 128], mode='bilinear', align_corners=True),
                cache_feats=True
            )
        min_std = torch.tensor(0.01, device=device)
        readout_feats = [(model.features[i] - param_dict[i][0]) / torch.max(param_dict[i][1], min_std)
                         for i in range(N+1)]

        """# 2. Information Restriction: """
        X_id_restrict = torch.zeros_like(X_id).to(device)  # [2*B, 512]
        Xt_feats, Xs_feats, X_lambda = [], [], []
        Xt_lambda, Xs_lambda = [], []
        Rs_params, Rt_params = [], []

        for i in range(N):
            R = model.features[i]  # [2*B, Cr, Hr, Wr]  model.module.features[i] if PARALLEL else
            Z, lambda_, info = getattr(iib, f'iba_{i}')(
                R, readout_feats,
                m_r=param_dict[i][0], std_r=param_dict[i][1],
                active_neurons=param_dict[i][2]
            )

            X_id_restrict += model.restrict_forward(Z, i)  # [2*B, 512]  model.module.restrict_forward(Z, i) if PARALLEL else
            Rs, Rt = R[:B], R[B:]
            lambda_s, lambda_t = lambda_[:B], lambda_[B:]

            m_s = torch.mean(Rs, dim=0)  # [C, H, W]
            std_s = torch.mean(Rs, dim=0)
            Rs_params.append([m_s, std_s])

            eps_s = torch.randn(size=Rt.shape).to(Rt.device) * std_s + m_s
            feat_t = Rt * (1. - lambda_t) + lambda_t * eps_s
            Xt_feats.append(feat_t)  # only related with lambda

            m_t = torch.mean(Rt, dim=0)  # [C, H, W]
            std_t = torch.mean(Rt, dim=0)
            Rt_params.append([m_t, std_t])
            Xt_lambda.append(lambda_t)

        X_id_restrict /= float(N)
        Xs_id, Xt_id = X_id_restrict[:B], X_id_restrict[B:]

        """# 3. Inference: """
        Xt_attr, Xt_attr_lamb = decoder(Xt, Xt_feats, lambs=Xt_lambda, use_lambda=True)
        Y = G(Xs_id, Xt_attr, Xt_attr_lamb)
        print(Y.shape)

        Y_id_gt = model(
            F.interpolate(Y[:, :, 74:950, 74:950], size=[128, 128], mode='bilinear', align_corners=True),
            cache_feats=True
        )
        msg = "Xs and Xt: %.4f" % torch.cosine_similarity(Xt_id, Xs_id, dim=1).mean().detach().cpu().numpy()
        msg += "Y and Xt: %.4f" % torch.cosine_similarity(Xt_id, Y_id_gt, dim=1).mean().detach().cpu().numpy()
        msg += "Y and Xs: %.4f" % torch.cosine_similarity(Xs_id, Y_id_gt, dim=1).mean().detach().cpu().numpy()

        batch_time = time.time() - start_time
        logger.info(msg)
        logger.info('Batch_time: %.5f' % batch_time)
        logger.info("===========================")

        I = [Xs, Xt, Y]
        image = make_image(I, args.show_size)
        cv2.imwrite(os.path.join(save_dir, img_name), image.transpose([1, 2, 0]),
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 0])  # [int(cv2.IMWRITE_JPEG_QUALITY), 100]


if __name__ == '__main__':
    inference_date = str(datetime.strptime(
        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
        "%a, %d %b %Y %H:%M:%S") + timedelta(hours=12)).split(' ')[0]

    p = argparse.ArgumentParser()
    p.add_argument('-src', '--source_dir', type=str, required=True,
                   help='DIR-OF-SOURCE-IMAGES-1024x1024')
    p.add_argument('-tar', '--target_dir', type=str, required=True,
                   help='DIR-OF-TARGET-IMAGES-1024x1024')
    p.add_argument('-save', '--save_dir', type=str, default='./results_1024')
    args = p.parse_args()

    assert os.path.exists(args.source_dir) and os.path.exists(args.target_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = os.path.join(args.save_dir, inference_date)
    os.makedirs(save_dir, exist_ok=True)

    SRCSET_PATH_LIST_1024 = [args.source_dir, ]
    TARSET_PATH_LIST_1024 = [args.target_dir, ]
    """
        If you have many source and target images datasets, use this:
        SRCSET_PATH_LIST_1024 = [
            'DIR-OF-SOURCE-IMAGES-1024x1024',
            '..',
        ]
    
        TARSET_PATH_LIST_1024 = [
            'DIR-OF-TARGET-IMAGES-1024x1024',
            '..',
        ]
    """

    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    handler = logging.FileHandler(filename=os.path.join(args.save_dir, f'train_{inference_date}.log'))
    train_formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    handler.setFormatter(train_formatter)
    logger.addHandler(handler)

    """ load config: """
    B = 1
    threshold = 0.01
    crop_size = (1024, 1024)
    dataset = FaceAlign(TARSET_PATH_LIST_1024, tar_random=False,
                        src_path_list=SRCSET_PATH_LIST_1024, crop_size=crop_size)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, num_workers=0, drop_last=True)
    path = './checkpoints_1024/w_kernel_smooth/ckpt_ks_*_1024.pth'
    pathG = path.replace('*', 'G')
    pathE = path.replace('*', 'E')
    pathI = path.replace('*', 'I')

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    model = Backbone128(50, 0.6, 'ir_se').eval().to(device)
    state_dict = torch.load('./model_128_ir_se50.pth', map_location=device)
    model.load_state_dict(state_dict, strict=True)

    G = AII1024().eval().to(device)  # if args.convex == 'no' else GeneratorLambda512pixConvex().train().to(device)
    G.load_state_dict(torch.load(os.path.join(args.root, pathG), map_location=device), strict=False)
    print("Successfully load G!")

    decoder = UnetDecoder1024().eval().to(device)
    decoder.load_state_dict(torch.load(os.path.join(args.root, pathE), map_location=device), strict=False)
    print("Successfully load decoder!")

    N = 10
    print(f"Using {N} perceptual features!")
    _ = model(torch.rand(B, 3, 128, 128).to(device), cache_feats=True)
    _readout_feats = model.features[:(N + 1)]  # one layer deeper than the z_attrs needed
    in_c = sum(map(lambda f: f.shape[-3], _readout_feats))
    out_c_list = [_readout_feats[i].shape[-3] for i in range(N)]
    iib = IIB(in_c, out_c_list, device, smooth=True, kernel_size=1).eval()
    iib.load_state_dict(torch.load(pathI, map_location=device), strict=False)
    print("Successfully load iib!")

    with torch.no_grad():
        inference_1024(save_dir)