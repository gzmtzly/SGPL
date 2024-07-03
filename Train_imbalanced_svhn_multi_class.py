import time, math, Model
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from collections import Counter
import sklearn.metrics as metrics
import torch.nn.functional as F
import Loss_function

batch_size = 100
num_class = 10
Epochs = 200

maj_class = [0, 1, 2, 3, 4]
data_num = '5000_100'
# data_num = '100_1000_3000_5000'
#
abs_path = './Linear_Imbalanced_svhn/'
abs_path_train = abs_path + data_num
print('maj_class:', maj_class)


train_data_path = abs_path_train + '/imbalanced_x.npy'
train_label_path = abs_path_train + '/imbalanced_y.npy'
test_data_path = abs_path + 'eval_x.npy'
test_label_path = abs_path + 'eval_y.npy'

# Loss_fun = "CE"
# Loss_fun = "FL"
# Loss_fun = 'CFL'
# Loss_fun = "ASL"
# Loss_fun = 'GPPE'
Loss_fun = 'SGPL'

train_times = 1

if Loss_fun == 'CE':
    Loss = nn.CrossEntropyLoss().cuda()
    save_confu_path = abs_path + data_num + '/result/result_CE/' + str(train_times) + '/'
elif Loss_fun == 'FL':
    Loss = Loss_function.Focal_Loss().cuda()
    save_confu_path = abs_path + data_num + '/result/result_FL/' + str(train_times) + '/'
elif Loss_fun == 'CFL':
    Loss = Loss_function.Cyclical_FocalLoss(epochs=Epochs).cuda()
    save_confu_path = abs_path + data_num + '/result/result_CFL/' + str(train_times) + '/'
elif Loss_fun == 'ASL':
    Loss = Loss_function.ASLSingleLabel().cuda()
    save_confu_path = abs_path + data_num + '/result/result_ASL/' + str(train_times) + '/'
elif Loss_fun == 'GPPE':
    Loss = Loss_function.GPPE_Multi_Class().cuda()
    save_confu_path = abs_path + data_num + '/result/result_GPPE/' + str(train_times) + '/'
elif Loss_fun == 'Im_GPPE':
    Loss = Loss_function.SGPL().cuda()
    save_confu_path = abs_path + data_num + '/result/result_SGPL/' + str(train_times) + '/'

print('Loss:', Loss)
print('save_confu_path:', save_confu_path)

print('train_data_path:', train_data_path)
print('test_data_path:', test_data_path)

train_data = np.load(train_data_path)
train_label = np.load(train_label_path)
print('train_label:', Counter(train_label))

test_data = np.load(test_data_path)
test_label = np.load(test_label_path)
print('test_label:', Counter(test_label))

size_per_class = [Counter(train_label)[i] for i in range(len(Counter(train_label)))]
print('size_per_class:', size_per_class)
frequency = size_per_class / np.sum(size_per_class)
K_ci = 1 / (np.max(frequency) - np.min(frequency)) * (frequency - np.min(frequency))
print('K_ci:', K_ci)

ssl_data_seed = 1
rng_data = np.random.RandomState(ssl_data_seed)

train_inds = rng_data.permutation(train_data.shape[0])
train_data = train_data[train_inds]
train_label = train_label[train_inds]

test_inds = rng_data.permutation(test_data.shape[0])
test_data = test_data[test_inds]
test_label = test_label[test_inds]

num_batch_train = math.ceil(train_data.shape[0] / batch_size)
test_num_bathces = math.ceil(test_data.shape[0] / batch_size)
print('test_num_bathces：', test_num_bathces)

device = torch.device('cuda')
model = Model.Model_Cifar10(num_class).to(device=device)

maj_min_express = [0 if train_label[i] in maj_class else 1 for i in range(len(train_label))]
maj_min_express = Variable(torch.from_numpy(np.array(maj_min_express)).long()).cuda()
maj_min_express = maj_min_express.view(-1, 1)

learning_rate = 2e-4
lr = 2e-4
A = []
if __name__ == "__main__":
    for epoch in range(Epochs):
        print('epoch:', epoch)

        if epoch > 80:
            lr = learning_rate * (((Epochs - epoch) // 10) * 0.1 + 0.1)

        print('learning_rate:', lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
        t1 = time.time()
        for iteration in range(num_batch_train):
            from_l_c = iteration * batch_size
            to_l_c = (iteration + 1) * batch_size

            data = train_data[from_l_c:to_l_c]
            label = train_label[from_l_c:to_l_c]
            maj_min_express_ = maj_min_express[from_l_c:to_l_c]

            data = Variable(torch.from_numpy(data).float()).cuda()
            label = Variable(torch.from_numpy(label).long()).cuda()

            # 
            optimizer.zero_grad()

            output_label = model(data)
            if Loss_fun == 'GPPE':
                loss = Loss(output_label, label, maj_min_express_)
            elif Loss_fun == 'CFL':
                loss = Loss(output_label, label, epoch)
            elif Loss_fun == 'SGPL':
                loss = Loss(output_label, label, K_ci)
            else:
                loss = Loss(output_label, label)

            loss.backward()
            optimizer.step()

        print('Train_time:', time.time() - t1)
        ground_truth_valid = []
        label_valid = []
        probility_predicted = []
        if epoch % 1 == 0:
            for iteration in range(test_num_bathces):
                from_l_c = iteration * batch_size
                to_l_c = (iteration + 1) * batch_size

                batch_data = test_data[from_l_c:to_l_c]
                data = Variable(torch.from_numpy(batch_data).float()).cuda()

                batch_label = test_label[from_l_c:to_l_c]

                output = model(data)
                predict_pro = F.softmax(output, dim=1)
                pi_1_cpu = predict_pro.data.cpu().numpy()

                label_batch = torch.max(output, 1)[1].data

                label_batch = np.int32(label_batch.cpu().numpy())
                label_valid.extend(label_batch)
                ground_truth_valid.extend(batch_label)
                probility_predicted.extend(pi_1_cpu)

        Confu_matir = confusion_matrix(ground_truth_valid, label_valid, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        Acc = metrics.accuracy_score(ground_truth_valid, label_valid)
        A.append(Acc)
        np.save(save_confu_path + 'Acc.npy', A)
        print(Confu_matir)
        print('ACC:', Acc)

        if epoch > 50:
            np.save(save_confu_path + 'Confu_matir_' + str(epoch) + '.npy', Confu_matir)
            np.save(save_confu_path + 'predicted_probility_' + str(epoch) + '.npy', np.array(probility_predicted))
            np.save(save_confu_path + 'target_' + str(epoch) + '.npy', np.array(ground_truth_valid))
