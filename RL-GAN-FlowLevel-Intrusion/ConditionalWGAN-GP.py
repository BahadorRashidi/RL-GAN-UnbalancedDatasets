import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import time
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from models import *
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import StratifiedShuffleSplit

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MyDataset(Dataset):
    def __init__(self, data, label, maxmin):
        self.maxmin = maxmin
        if self.maxmin:
            self.data = self.maxmin.fit_transform(data)
        else:
            self.data = data
        self.data = (self.data - 0.5) / 0.5
        self.data = torch.FloatTensor(self.data)
        self.label = torch.LongTensor(label)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # data = torch.FloatTensor(data)
        # label = torch.LongTensor(self.label[idx])

        return self.data[idx], self.label[idx]

class Network(nn.Module):
    def __init__(self, input_dim, hidden, out_dim):
        super().__init__()
        # self.input_dim = input_dim
        # self.out_dim = out_dim
        self.linear = nn.Sequential(nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            # nn.Dropout(0.5)
            ))

    def forward(self, x):
        out = self.linear(x)
        return out

class Trainer():
    def __init__(self, vt, train, label, num, class_size, embedding_dim, batch_size, latent_size=2, device='cuda', lr=0.0001, num_workers=1, val_set=None, val_label=None):
        self.dataset = MyDataset(train, label, MinMaxScaler())
        self.val_set = MinMaxScaler().fit_transform(val_set)
        # print(self.val_set)
        self.val_label = val_label
        # self.val_data = MyDataset(self.val_set, self.val_label, None)


        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # self.data_loaderfor_val = torch.utils.data.DataLoader(self.val_data, batch_size = 64, shuffle = True, num_workers = num_workers)
        self.batch_size = batch_size
        self.device = device
        self.class_size = class_size
        self.num = num
        self.lrmodel = Network(10, 40, 5).cuda()
        state = torch.load("model_pca10.pth")
        self.lrmodel.load_state_dict(state)
        #define models
        self.latent_size = latent_size
        self.gen = Generator(vt, self.latent_size, class_size, embedding_dim).to(device)
        if vt:
            self.dis = Discriminator(class_size, embedding_dim).to(device)
            self.optimizer_g = optim.RMSprop(self.gen.parameters(), lr=lr)
            self.loss_func = nn.BCELoss().to(device)
            self.optimizer_d = optim.RMSprop(self.dis.parameters(), lr=lr)
        else:
            self.gen = self.load_model(self.gen, "generator_.pth")
            # print(self.gen.state_dict())
            # input()
            self.optimizer_g = optim.RMSprop(filter(lambda p: p.requires_grad, self.gen.parameters()), lr=lr)


    def load_model(self, model, path):
        model_dict = model.state_dict()#
        model_state = torch.load(path)
        new_dict = {k: v for k, v in model_state.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        # print("load finish")
        # input()
        return model


    def gradient_penalty(self, real_samples, fake_samples, labels):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1))).to(self.device)
        # print(alpha.shape)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        # print(interpolates.shape)
        d_interpolates = self.dis(interpolates, labels)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        fake.requires_grad = False
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradients = gradients[0].view(gradients[0].size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_noise(self, epochs):
        print('-----------------------')
        SAMPLE_SIZE = self.num.sum()
        fixed_noise = torch.load('noise.pth')
        # print(fixed_noise)
        # input()
        fixed_labels = torch.zeros(SAMPLE_SIZE, 4)
        label_for_test = np.zeros(SAMPLE_SIZE)
        base = 0
        la = 0
        m = 0
        acc = []
        loss_ = []
        # print(self.val_set)
        # print('--------------')
        val_set = torch.FloatTensor(self.val_set).cuda()
        for i in self.num:
            fixed_labels[base: base + i, la] = 1
            label_for_test[base:base + i] = la
            la += 1
            base = base + i

        label_for_test = torch.LongTensor(label_for_test).cuda()
        normal_loss = []
        dos_loss  = []
        probe_loss = []
        r2l_loss = []
        # print(self.val_label)
        # label_for_test = torch.LongTensor(self.val_label).cuda()
        for epoch in tqdm(range(10000)):
            # print(epoch)
            self.optimizer_g.zero_grad()
            # print(fixed_noise)
            # print(label_for_test)
            val_out = self.gen(fixed_noise, label_for_test)
            # print(val_out)
            # input()
            val_out = (val_out * 0.5) + 0.5

            loss = torch.mean((val_out - val_set)**2)
            # ind = np.where(self.val_label == 2)
            # print(ind[0])
            # print(self.val_set[ind[0]].shape)
            # input()
            # print(self.val_set.max())
            base = 0
            for i in range(0, 4):
                ind = np.where(self.val_label == i)
                dis = abs(val_set[ind[0]] - val_out[base:base + len(ind[0])]) ** 2

                base += len(ind[0])
                dis = dis.mean()
                if i==2:
                    normal_loss.append(dis.item())
            # print(self.num[1], self.num[2])
            # normal_loss.append(torch.mean((val_out[self.num[0]:self.num[0]+self.num[1]]-val_set[self.num[0]:self.num[0]+self.num[1]])**2).item())
                # fixed_labels[base: base + i, la] = 1
                # label_for_test[base:base + i] = la
                # la += 1
                # base = base + i

            loss.backward()
            self.optimizer_g.step()
            # print(loss)
            loss_.append(loss.item())

            # if epoch > 0:
            #     print(loss.item())
        print(normal_loss[0])    # input()
        plt.plot(np.arange(10000), normal_loss)
        plt.show()
    def train(self, epochs, saved_image_directory, saved_model_directory):
        start_time = time.time()
        # setup_seed(0)

        gen_loss_list = []
        dis_loss_list = []
        was_loss_list = []
        # print('--------')
        store_information = pd.DataFrame(columns=['acc', 'iteration', 'class'])
        store_loss = pd.DataFrame(columns=['dis', 'iteration', 'class'])

        lmbda_gp = 10
        SAMPLE_SIZE = self.num.sum()
        fixed_noise = torch.FloatTensor(SAMPLE_SIZE, 2).normal_(0, 1).cuda()
        torch.save(fixed_noise, "noise.pth")
        print(fixed_noise)
        # tensor([[-0.0245, 0.6291],
        #         [-0.2760, 0.0105],
        #         [1.1604, 0.1375],
        #         ...,
        #         [0.9687, 0.6736],
        #         [-0.4794, -1.2838],
        #         [1.4275, -0.6287]], device='cuda:0')
        # input()
        # print(fixed_noise)
        # input()
        fixed_labels = torch.zeros(SAMPLE_SIZE, 4)
        # fixed_labels_onehot = torch.FloatTensor(SAMPLE_SIZE, NUM_LABELS)

        # label = torch.FloatTensor(BA)
        # one_hot_labels = torch.FloatTensor(BATCH_SIZE, NUM_LABELS)
        label_for_test = np.zeros(SAMPLE_SIZE)
        base = 0
        la = 0
        m = 0
        acc = []
        for i in self.num:
            fixed_labels[base: base + i, la] = 1
            label_for_test[base:base + i] = la
            la += 1
            base = base + i
        label_for_test = torch.LongTensor(label_for_test).cuda()

        for epoch in range(250):
            gen_loss = 0
            dis_loss = 0
            cur_time = time.time()
            for images, labels in self.data_loader:
                self.gen.train()
                self.dis.train()
                # print(images.shape)
                b_size = len(images)
                #train Discriminator with Wasserstein Loss
                self.optimizer_d.zero_grad()

                #fake loss
                z = torch.randn(b_size, self.latent_size).to(self.device)
                # print(z)
                fake_images = self.gen(z, labels.to(self.device))
                fake_pred = self.dis(fake_images, labels.to(self.device))
                d_loss_fake = torch.mean(fake_pred)

                #real loss
                real_pred = self.dis(images.to(self.device), labels.to(self.device))
                d_loss_real = -torch.mean(real_pred)

                gp = self.gradient_penalty(images.to(self.device), fake_images, labels.to(self.device))

                d_loss = d_loss_fake - d_loss_real
                was_loss = (d_loss_fake + d_loss_real) + lmbda_gp*gp
                was_loss.backward()
                self.optimizer_d.step()

                dis_loss += d_loss.item()/b_size

               
                #train Generator
                self.optimizer_g.zero_grad()

                z = torch.randn(b_size, self.latent_size).to(self.device)
                fake_images = self.gen(z, labels.to(self.device))
                fake_pred = self.dis(fake_images, labels.to(self.device))
                g_loss = -torch.mean(fake_pred)
                g_loss.backward()
                self.optimizer_g.step()

                gen_loss += g_loss.item()/b_size

            cur_time = time.time() - cur_time

            print('Epoch {},    Gen Loss: {:.4f},   Dis Loss: {:.4f},   Was Loss: {:.4f}'.format(epoch, gen_loss, dis_loss, was_loss))
            print('Time Taken: {:.4f} seconds. Estimated {:.4f} hours remaining'.format(cur_time, (epochs-epoch)*(cur_time)/3600))
            gen_loss_list.append(gen_loss)
            dis_loss_list.append(dis_loss)
            was_loss_list.append(was_loss)
            self.gen.eval()
            self.dis.eval()
            # print('------------------', label_for_test)
            # gf = torch.load("generator250.pth")
            # self.gen.load_state_dict(gf)
            test_out = self.gen(fixed_noise, label_for_test)
            # print(label_for_test)
            # print(test_out)

            test_out = (test_out * 0.5) + 0.5
            # print(test_out)
            # input()
            pred = self.lrmodel(test_out)
            pred = torch.argmax(pred, dim=1)

            acc1 = torch.eq(pred, label_for_test).sum().item() / len(label_for_test)
            pred = pred.data.cpu().numpy()
            test_out = test_out.data.cpu().numpy()
            label_for_test = label_for_test.data.cpu().numpy()
            # print(confusion_matrix(label_for_test, pred))
            s = precision_score(label_for_test, pred, average=None)
            classs = ['Normal', 'Dos', 'Probe', 'R2L']
            base = 0
            sorted_id = sorted(range(len(self.val_label[:, 0])), key=lambda k: self.val_label[:, 0][k], reverse=False)
            # v = self.val_set[sorted_id]
            # print(np.mean((v-test_out)**2))
            for i in range(0, 4):
                ind = np.where(self.val_label == i)
                # print(ind[0])
                # print(self.val_set[ind[0]].shape)
                # input()
                # print(self.val_set.max())
                dis = abs(self.val_set[ind[0]] - test_out[base:base+len(ind[0])])**2

                base+=len(ind[0])# print(dis)
                # input()
                dis = dis.mean()
                # print(dis)
                # dis = self.val_set[ind] -
                store_loss = store_loss.append({'dis':dis, 'iteration':epoch, 'class':classs[i]},ignore_index=True)
            # input()

            # print(precision_score(label_for_test, pred, average=None))
            store_information = store_information.append({'acc':acc1, 'iteration':epoch, 'class':'Overall'},ignore_index=True)
            store_information = store_information.append({'acc':s[0], 'iteration':epoch, 'class': 'Normal'},ignore_index=True)
            store_information = store_information.append({'acc': s[1], 'iteration': epoch, 'class': 'Dos'},ignore_index=True)
            store_information = store_information.append({'acc': s[2], 'iteration': epoch, 'class': 'Probe'},ignore_index=True)
            # store_information = store_information.append({'acc': s[3], 'iteration': epoch, 'class': 'R2L'},ignore_index=True)
            label_for_test = torch.LongTensor(label_for_test).cuda()
            acc.append(acc1)
            # print(acc1)

            #show samples
            # labels = torch.LongTensor(np.arange(10)).to(self.device)
            # z = torch.randn(10, self.latent_size).to(self.device)
            # sample_images = self.gen(z, labels)
            #
            # #save models to model_directory
            # torch.save(self.gen.state_dict(), saved_model_directory + '/generator_{}.pt'.format(epoch))
            # torch.save(self.dis.state_dict(), saved_model_directory + '/discriminator_{}.pt'.format(epoch))
            #
            # image_grid = torchvision.utils.make_grid(sample_images.cpu().detach(), nrow=5, normalize=True)
            # _, plot = plt.subplots(figsize=(12, 12))
            # plt.axis('off')
            # plot.imshow(image_grid.permute(1, 2, 0))
            # plt.savefig(saved_image_directory + '/epoch_{}_checkpoint.jpg'.format(epoch), bbox_inches='tight')
        # plt.plot(np.arange(50), acc)
        torch.save(self.gen.state_dict(), "generator_.pth")
        # print(test_out)
        plt.figure()
        sns.lineplot(data=store_information, x='iteration', y = 'acc', hue = 'class')
        plt.savefig("Onoiseacc_b256_250epoch_withoutinitN.svg")
        plt.figure()
        sns.lineplot(data=store_loss, x='iteration', y ='dis', hue = 'class')
        plt.savefig("NOnoiseloss_b256_250epoch_withoutinit.svg")
        finish_time = time.time() - start_time
        print('Training Finished. Took {:.4f} seconds or {:.4f} hours to complete.'.format(finish_time, finish_time/3600))
        return gen_loss_list, dis_loss_list



def getdataset(path1, path2, pca_dimension, add_noise = False):
    print(add_noise)
    train_set = pd.read_csv(path1, index_col=0)
    # print(trai
    # input()
    train_label = pd.read_csv(path2, index_col=0)
    # train_set = train_set.join(train_label)
    # if add_noise:
    #     R2L = train_set[train_set['class'] == 3]
    #     print(R2L)
    #     input()
    ss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, train_size=0.9, random_state = 33)
    ss2 = StratifiedShuffleSplit(n_splits=9, test_size=0.111, train_size=0.889, random_state= 33)
    dele = list(np.where(train_label['class'] == 4))


    train_set = train_set.drop(index = dele[0]).to_numpy()
    train_label = train_label.drop(index = dele[0]).to_numpy()
        # num = np.array(list(dict(Counter(train_label['class'])).alues()))
    # num = (num/num.sum()*int(num.sum()/10))
    # num = [int(i) for i in num]
    # num = sorted(num, reverse=True)
    print(train_set.shape)
    print(train_label.shape)
    # input()
    for tra, val in ss.split(train_set, train_label):
        # print(len(tra), len(val))

        train_set1, train_label1 = train_set[tra], train_label[tra]
        for tra_, val_ in ss2.split(train_set1, train_label1):
            val_set, val_label = train_set1[val_], train_label1[val_]
            train_set1, train_label1 = train_set1[tra_], train_label1[tra_]
            break
        # input()
        test_set, test_label = train_set[val], train_label[val]
        break
    num = []
    for i in range(4):
        num.append(len(val_label[val_label == i]))
    # print(val_label[val_label == 0])
    # print(num)
    # input()
    print(train_set1.shape, val_set.shape)
    # input()
    maxmin = MinMaxScaler()
    train_set = maxmin.fit_transform(train_set1)
    val_set = maxmin.fit_transform(val_set)
    # print(train_set.shape)
    pca = PCA(n_components=pca_dimension, random_state=33)
    gac = pca.fit(train_set)
    train_set = gac.transform(train_set)
    # print(train_set)
    if add_noise:
        ind = np.where(train_label1 == 3)
        # print(len(ind[0]))
        gua = np.random.randn(len(ind[0]), pca_dimension)*0.01
        # print(gua.shape)
        temp = train_set[ind[0]] + gua
        # temp = np.clip(temp, 0, 1)
    # print(temp)
        add_label = np.ones((len(ind[0]), 1))*3

        train_set = np.vstack((train_set, temp))
        train_label1 = np.vstack((train_label1, add_label))
    # print(train_label1.shape)
    # print(train_set.shape)
    # input()
    val_set = gac.transform(val_set)
    # print(train_set.shape)

    # input()

    return train_set,  train_label1, np.array(num), val_set, val_label

import random
def spilt(ratio, X_train, y_train):
    index = np.arange(0, len(X_train))
    random.shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]
    num_of_train = int(ratio * len(X_train))
    return X_train[0:num_of_train], y_train[0:num_of_train], X_train[num_of_train:], y_train[num_of_train:]

from Preprocessing import Preprocessing
def main():
    setup_seed(0)
#     parser = argparse.ArgumentParser(description='Hyperparameters for training GAN')
#     #hyperparameter loading
#     parser.add_argument('--data_directory', type=str, default='data', help='directory to MNIST dataset files')
#     parser.add_argument('--saved_image_directory', type=str, default='data/saved_images', help='directory to where image samples will be saved')
#     parser.add_argument('--saved_model_directory', type=str, default='saved_models', help='directory to where model weights will be saved')
#     parser.add_argument('--class_size', type=int, default=4, help='number of unique classes in dataset')
#     parser.add_argument('--embedding_dim', type=int, default=4, help='size of embedding vector')
#     parser.add_argument('--batch_size', type=int, default=256, help='size of batches passed through networks at each step')
#     parser.add_argument('--latent_size', type=int, default=100, help='size of gaussian noise vector')
#     parser.add_argument('--device', type=str, default='cuda', help='cpu or gpu depending on availability and compatability')
#     parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of models')
#     parser.add_argument('--num_workers', type=int, default=4, help='workers simultaneously putting data into RAM')
#     parser.add_argument('--epochs', type=int, default=50, help='number of iterations of dataset through network for training')
#     args = parser.parse_args()

    Pre = Preprocessing('KDDTrain+.txt')
    train_data, train_label = Pre.deal_with_lines()
    X_train, y_train, X_val, y_val = spilt(0.8, train_data, train_label)
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    print(np.max(train_data), X_train.shape)
    # input()
    # input()
    PCA_DIMENSION = 10
    train_set, train_label, num, val_set, val_label = getdataset("/home/binyang/PycharmProjects/AE/train_embedding.csv","/home/binyang/PycharmProjects/AE/train_label.csv", PCA_DIMENSION)

    data_dir = args.data_directory
    saved_image_dir = args.saved_image_directory
    saved_model_dir = args.saved_model_directory
    class_size = args.class_size
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    latent_size = args.latent_size
    device = args.device
    lr = args.lr
    num_workers = args.num_workers
    epochs = args.epochs
    gan = Trainer(False, train_set, train_label, num, class_size, embedding_dim, batch_size, 2, device, lr, num_workers, val_set, val_label)
    # gen_loss_lost, dis_loss_list = gan.train(epochs, saved_image_dir, saved_model_dir)
    gan.train_noise(epochs)

if __name__ == "__main__":
    main()
