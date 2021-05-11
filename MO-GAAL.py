from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import argparse
from datetime import date
from datetime import datetime
import glob, os
import h5py
import pdb
from sklearn import preprocessing
from sklearn import metrics 


def parse_args():
    parser = argparse.ArgumentParser(description="Run MO-GAAL.")
    parser.add_argument('--path', nargs='?', default='Data/cic_ids_2017',
                        help='Input data path.')
    parser.add_argument('--stop_epochs', type=int, default=20,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate of model.')
    parser.add_argument('--decay', type=float, default=1e-6,
                        help='Decay.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')
    parser.add_argument('--all_data', default="",
                        help='Take all files in path')
    parser.add_argument('--data_splitted', default="",
                        help='Data are already splitted')
    parser.add_argument('--k', default=1,
                        help='number of sub generators')

    args = parser.parse_args()

    dict_args = {
        'path'         : args.path,
        'stop_epochs'  : args.stop_epochs,
        'lr'           : args.lr,
        'k'            : int(args.k),
        'decay'        : args.decay,
        'momentum'     : args.momentum,
        'all_data'     : bool(args.all_data),
        'data_splitted': bool(args.data_splitted)
    }

    return dict_args

# Generator
def create_generator(latent_size):
    gen = Sequential()
    gen.add(Dense(latent_size, input_dim=latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
    gen.add(Dense(latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
    latent = Input(shape=(latent_size,))
    fake_data = gen(latent)
    return Model(latent, fake_data)

# Discriminator
def create_discriminator():
    dis = Sequential()
    dis.add(Dense(math.ceil(math.sqrt(data_size)), input_dim=latent_size, activation='relu', kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    dis.add(Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    data = Input(shape=(latent_size,))
    fake = dis(data)
    return Model(data, fake)

# Load data
def load_data():
    if all_data:
        os.chdir(path_data)
    else:
        os.chdir("/".join(path_data.split("/")[:-1]))

    df = pd.DataFrame()

    labels = None
    if all_data:
        for file in glob.glob("*.h5"):
            features = h5py.File(file, mode='r')['vectors']
            np_array = np.array(features)
            df = df.append(pd.DataFrame(np_array), ignore_index=True)

        csvs = glob.glob("*.csv")

        assert(len(csvs) == 1)

        labels = pd.read_table(csvs[0], sep=',')["Label"]
    else:
        df = pd.read_table('{path}'.format(path = path_data.split("/")[-1]), sep=',', header=None)
        df.pop(0)
        labels = df.pop(1)

    
    scaler = preprocessing.MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df))

    df = df.sample(frac=1).reset_index(drop=True)
    return df.values, labels.values

def load_data_splitted(path_data):
    path_data_train, path_data_test = path_data 
    df_train = pd.read_csv(path_data_train)
    df_test = pd.read_csv(path_data_test)

    label_train = pd.read_csv(path_data_train.replace(".csv", "") + "_label.csv")
    label_test = pd.read_csv(path_data_test.replace(".csv", "") + "_label.csv")

    return df_train.values, label_train.values, df_test.values, label_test.values



# Plot loss history
def plot(train_history, title, field, args, label):
    plt.style.use("ggplot")
    fig = plt.figure()
    plt.plot(np.arange(0, len(train_history[field])), train_history[field], color="red", label=label)
    if field + "_test" in train_history:
      plt.plot(np.arange(0, len(train_history[field + "_test"])), train_history[field + "_test"], color='blue', label=field + " test")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel(title)
    plt.legend(loc="upper right")
    ax = fig.get_axes()[0]
    #plt.setp(ax.get_yticklabels(), visible=False)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    name = "res/vect_res_{}_db_{}_LR_{}_momentum_{}_decay_{}_{}_{}.png".format(field, args['path'].replace("/", "-"), args['lr'], args['momentum'], args['decay'], date.today(), current_time.replace(":", "-"))

    print("on sav dans le fichier: " + name)
    plt.savefig(name)

def count_occ_eq_and_inf(value, tab, start_index):
    nbr_occ = 0
    index_first_occ = None
    for i in range(start_index, len(tab)):
        if tab[i] == value:
            if index_first_occ == None:
                index_first_occ = i
            nbr_occ += 1
        elif tab[i] > value:
            index_first_occ = index_first_occ if index_first_occ != None else i
            return index_first_occ, nbr_occ, index_first_occ
    # si on croise pas de > donc quand tout le tab est < et/ou ==
    if index_first_occ == None:
        return len(tab), 0, len(tab)
    else:
        return len(tab) - nbr_occ, nbr_occ, len(tab) 

def calc_auc(train_history, field, to_print, discriminator, data_x, data_y):
    # Detection result
    '''p_value = discriminator.predict(data_x)
    p_value = pd.DataFrame(p_value)
    data_y = pd.DataFrame(data_y)
    result = np.concatenate((p_value,data_y), axis=1)
    result = pd.DataFrame(result, columns=['p', 'y'])
    result = result.sort_values('p', ascending=True)

    # Calculate the AUC
    inlier_parray = result.loc[lambda df: (df.y == 0.0) | (df.y == "nor")]["p"].values
    outlier_parray = result.loc[lambda df: (df.y == 1.0) | (df.y == "out")]["p"].values
    sum = 0.0
    start_index = 0
    for i in inlier_parray:
        nbr_inf, nbr_eq, st_i = count_occ_eq_and_inf(i, outlier_parray, start_index)
        start_index = st_i
        sum += nbr_inf
        sum += (nbr_eq * 0.5)
    acc = (sum / (len(inlier_parray) * len(outlier_parray)))'''

    pred = discriminator.predict(data_x)
    _, _, thresholds = metrics.roc_curve(data_y, pred)

    acc_max = 0
    for thres in thresholds:
        y_pred = np.where(pred < thres, 1, 0)
        acc_tmp = metrics.accuracy_score(data_y, y_pred)
        if acc_tmp > acc_max:
            acc_max = acc_tmp
    
    print(to_print + "  " +"{:.4f}".format(acc_max))
    train_history[field].append(acc_max)

if __name__ == '__main__':
    train = True

    # initilize arguments
    args = parse_args()

    if not args['data_splitted']:
        data_x, data_y = load_data(args['path'], args['all_data']) # faut mettre le dossier, apres load_data se charge du reste
        rows = np.random.choice(data_x.shape[0], size=data_x.shape[0] // 10, replace=True)
        data_x_test = data_x[rows]
        data_x = data_x[~rows]
        data_y_test = data_y[rows]
        data_y = data_y[~rows]

    else:
        data_x, data_y, data_x_test, data_y_test = load_data_splitted(args['path'].split("%"))

    latent_size = data_x.shape[1]
    data_size = data_x.shape[0]
    print("The dimension of the training data :{}*{}".format(data_x.shape[0], data_x.shape[1]))

    if train:
        train_history = defaultdict(list)
        names = locals()
        epochs = args['stop_epochs'] * 3
        stop = 0
        k = args['k']

        # Create discriminator
        discriminator = create_discriminator()
        discriminator.compile(optimizer=SGD(lr=args['lr'], decay=args['decay'], momentum=args['momentum']), loss='binary_crossentropy')

        # Create k combine models
        for i in range(k):
            names['sub_generator' + str(i)] = create_generator(latent_size)
            latent = Input(shape=(latent_size,))
            names['fake' + str(i)] = names['sub_generator' + str(i)](latent)
            discriminator.trainable = False
            names['fake' + str(i)] = discriminator(names['fake' + str(i)])
            names['combine_model' + str(i)] = Model(latent, names['fake' + str(i)])
            names['combine_model' + str(i)].compile(optimizer=SGD(lr=args['lr'], decay=args['decay'], momentum=args['momentum']), loss='binary_crossentropy')

        # Start iteration
        for epoch in range(epochs):
            print('Epoch {} of {}'.format(epoch + 1, epochs))
            batch_size = min(500, data_size)
            num_batches = int(data_size / batch_size)

            for index in range(num_batches):
                print('\nTesting for epoch {} index {}:'.format(epoch + 1, index + 1))

                # Generate noise
                noise_size = batch_size
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))

                # Get training data
                data_batch = data_x[index * batch_size: (index + 1) * batch_size]

                # Generate potential outliers
                block = ((1 + k) * k) // 2
                for i in range(k):
                    if i != (k-1):
                        noise_start = int((((k + (k - i + 1)) * i) / 2) * (noise_size // block))
                        noise_end = int((((k + (k - i)) * (i + 1)) / 2) * (noise_size // block))
                        names['noise' + str(i)] = noise[noise_start : noise_end ]
                        names['generated_data' + str(i)] = names['sub_generator' + str(i)].predict(names['noise' + str(i)], verbose=0)
                    else:
                        noise_start = int((((k + (k - i + 1)) * i) / 2) * (noise_size // block))
                        names['noise' + str(i)] = noise[noise_start : noise_size]
                        names['generated_data' + str(i)] = names['sub_generator' + str(i)].predict(names['noise' + str(i)], verbose=0)

                # Concatenate real data to generated data
                for i in range(k):
                    if i == 0:
                        X = np.concatenate((data_batch, names['generated_data' + str(i)]))
                    else:
                        X = np.concatenate((X, names['generated_data' + str(i)]))
                Y = np.array([1] * batch_size + [0] * int(noise_size))

                # Train discriminator
                discriminator_loss = discriminator.train_on_batch(X, Y)
                train_history['discriminator_loss'].append(discriminator_loss)

                # Get the target value of sub-generator
                p_value = discriminator.predict(data_x)
                p_value = pd.DataFrame(p_value)
                for i in range(k):
                    names['T' + str(i)] = p_value.quantile(i/k)
                    names['trick' + str(i)] = np.array([float(names['T' + str(i)])] * noise_size)

                # Train generator
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))
                if stop == 0:
                    for i in range(k):
                        names['sub_generator' + str(i) + '_loss'] = names['combine_model' + str(i)].train_on_batch(noise, names['trick' + str(i)])
                        train_history['sub_generator{}_loss'.format(i)].append(names['sub_generator' + str(i) + '_loss'])
                else:
                    for i in range(k):
                        names['sub_generator' + str(i) + '_loss'] = names['combine_model' + str(i)].evaluate(noise, names['trick' + str(i)])
                        train_history['sub_generator{}_loss'.format(i)].append(names['sub_generator' + str(i) + '_loss'])

                generator_loss = 0
                for i in range(k):
                    generator_loss = generator_loss + names['sub_generator' + str(i) + '_loss']
                generator_loss = generator_loss / k
                train_history['generator_loss'].append(generator_loss)

                # Stop training generator
                if epoch +1  > args['stop_epochs']:
                    stop = 1

            # Calc auc train

            calc_auc(train_history, 'auc', "AUC", discriminator, data_x, data_y)

            # calc auc test Test

            calc_auc(train_history, 'auc_test', "AUC_test", discriminator, data_x_test, data_y_test)

        
    print("maintenant on affiche")
    plot(train_history, 'AUC', 'auc', args, 'train acc')
    plot(train_history, 'discriminator_loss', 'discriminator_loss', args, 'discri loss')
    plot(train_history, 'generator_loss', 'generator_loss', args, 'gene loss')
