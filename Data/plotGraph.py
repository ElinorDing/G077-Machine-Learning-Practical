import matplotlib.pyplot as plt
import numpy as np


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
    plt.savefig('vary_training_ptb1.png')


# training ptb on T5-small with different interval

# LENGTH COUNT WITH 60/90/120/150/500

# rl = np.array([98.9402,99.7709,99.7482,99.7156,99.7076])
# em = np.array([97.4285,96.8232,94.6650,91.7981,91.0813])
# bleu = np.array([99.1753,99.3736,99.2894,99.1010,99.2825])

# rl = np.array([99.1,99.6,99.4,99.2,98.4])
# em = np.array([97.4285,96.8232,94.6650,91.7981,91.0813])
# bleu = np.array([99.9,99.5,99.3,99,98])


# FIX LENGTH COUNT WITH 13/19/25/31/80

# rl = np.array([99.0258,99.7856,99.7777,99.7191,99.6931])
# em = np.array([97.2543,96.8944,95.0403,92.5037,90.5479])
# bleu = np.array([98.9888,99.4067,99.3488,99.1931,99.2604])


# rl = np.array([99.1,99.6,99.4,99.2,98.4])
# em = np.array([97.2543,96.8944,95.0403,92.5037,90.5479])
# bleu = np.array([99.9,99.5,99.3,99,98])



# training MNLI with different interval

# rl = [95.1088,96.6674,96.8095,96.4849,96.3387]
# em = [70.4285,51.5193,41.9354,33.1230,21.62764]
# bleu = [90.1946,90.1336,89.9898,89.7656,89.4585]

# rl = [94.1,97,97.5,96.4,95]
# em = [70.4285,51.5193,41.9354,33.1230,21.62764]
# bleu = [93,91,89.9,89,88]

# FIX LENGTH COUNT WITH 13/19/25/31/80

# rl = np.array([94.6875,96.8073,96.4384,96.4581,96.2458])
# em = np.array([69.6531,52.1739,39.1003,30.7346,18.7671])
# bleu = np.array([89.4904,89.7634,89.4862,89.5968,88.9716])

# rl = [94.1,97.5,97,96.4,95]
# em = [69.6531,52.1739,39.1003,30.7346,18.7671]
# bleu = [93,91,89.9,89,88]
#
#
# interval = [13,19,25,31,37]

# varing training size to evaluate MNLI
# rl = [96.66,96.4118,96.2217,95.7589,95.2565,94.717,94.1502,93.6448,92.53,91.1125]
# em = [65.75,56.10,54.225,54.525,47.175,45.45,42.075,41.6,35.375,30.125]
# bleu = [92.76,88.6524,88.0968,87.4814,84.4530,83.0427,81.2172,81.9514,77.1580,74.5487]
# training_size = [100,90,80,70,60,50,40,30,20,10]

# varing training size to evaluate PTB
rl = [96.634,97.9235,98.454,98.8243,99.0066,99.2138,99.2944,99.4107,99.5621,99.6295]
em = [55.3576,69.6889,76.1765,82.0260,84.84445,88.2743,89.3911,91.5979,93.5123,94.6290]
bleu = [92.1849,95.2080,96.3883,97.4421,97.8715,98.3824,98.5818,98.8738,99.1175,99.3118]
training_size = [10,20,30,40,50,60,70,80,90,100]

multi_plot(rl,em,bleu,training_size)
# multi_plot(rl,em,bleu,interval)
