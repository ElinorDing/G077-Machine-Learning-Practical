import matplotlib.pyplot as plt
import numpy as np 


def multi_plot(resnet, vgg, vit,interval):
    plt.plot(interval, resnet,'+-', label = 'RESNET')
    plt.plot(interval, vgg,'+-', label = 'VGG16')
    plt.plot(interval, vit,'+-', label = 'VIT')
    # X_axis = np.arange(len(interval))
    # plt.bar(X_axis-0.2, resnet,0.2, label = 'RESNET')
    # plt.bar(X_axis, vgg, 0.2, label = 'VGG16')
    # plt.bar(X_axis+0.2, vit, 0.2,label = 'VIT')

    # plt.suptitle('Evaluation with different intervals', fontsize = 15)
    # my_xticks = ['<=13','13 ~ 19','19 ~ 25','25 ~ 31','>31']
    plt.xticks(interval)
    # plt.ylim(88,100)
    plt.margins(x=0)
    # plt.legend(['ROUGLE-L','Exact Match','BLEU'])
    plt.legend()
    # plt.xlabel('Various Length interval')
    plt.xlabel('Data Size for Training (%)')
    plt.ylabel('Accuracy(%)')
    # plt.show()
    plt.savefig('vary_training_accuracy')



# varing training size to evaluate accuracy
resnet = [90.248, 93.0348,92.039,93.432]
vgg16 = [90.54,97.80,97.94,98.59]
vit = [97.51,97.76,98.08,98.62]
training_size = [25, 50, 75, 100]

# varing training size to evaluate loss
# resnet = [0.2875,0.2356,0.2084,0.1806]
# vgg16 = [0.3635,0.2584,0.1426,0.1066]
# vit = [0.2817,0.1970,0.1638,0.1511]
# training_size = [25, 50, 75, 100]

# # varing training size to evaluate f1
# resnet = [90.233,93.078,92.039,93.430]
# vgg16 = [26.59,22.88,26.75,24.38]
# vit = [97.52,97.75,98.08,98.63]
# training_size = [25, 50, 75, 100]

multi_plot(resnet,vgg16,vit,training_size)
