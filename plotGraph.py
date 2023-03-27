import matplotlib.pyplot as plt


def multi_plot(rl, em, bleu,interval):
    plt.plot(interval, rl,'+-', label = 'ROUGLE-L')
    plt.plot(interval, em,'+-', label = 'Exact Match')
    plt.plot(interval, bleu,'+-', label = 'BLEU')

    # plt.bar(interval, rl,10,color='#FBDD7E')
    # plt.bar(interval, em, 10, bottom=rl,color='#04D8B2')
    # plt.bar(interval, bleu, 10, bottom=rl+em,color='#7BC8F6')

    # plt.suptitle('Evaluation with different intervals', fontsize = 15)
    # my_xticks = ['<=13','13 ~ 19','19 ~ 25','25 ~ 31','>31']
    plt.xticks(interval)
    # plt.ylim(88,100)
    plt.margins(x=0)
    # plt.legend(['ROUGLE-L','Exact Match','BLEU'])
    plt.legend()
    # plt.xlabel('Various Length interval')
    plt.xlabel('Data Size for Training (%)')
    plt.ylabel('Examination Results (%)')
    # plt.show()
    # plt.savefig('vary_training_ptb1.png')



# varing training size to evaluate accuracy
resnet = [90.248, 93.0348,92.039,93.432]
vgg16 = []
vit = []
training_size = [25, 50, 75, 100]

# varing training size to evaluate loss
resnet = [27.0622,21.774,20.129,17.201]
vgg16 = []
vit = []
training_size = [25, 50, 75, 100]

# varing training size to evaluate f1
resnet = [90.233,93.078,92.039,93.430]
vgg16 = []
vit = []
training_size = [25, 50, 75, 100]

multi_plot(rl,em,bleu,training_size)
