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
    parser = argparse.ArgumentParser(description="Run SO-GAAL.")
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
                        help='Data are already splitted'),

    parser.add_argument('--auc_other', default="",
                        help='calc other auc')

    parser.add_argument('--to_shuffle', default="",
                        help='if data are not shuffled')

    args = parser.parse_args()

    dict_args = {
        'path'         : args.path,
        'stop_epochs'  : args.stop_epochs,
        'lr'           : args.lr,
        'decay'        : args.decay,
        'momentum'     : args.momentum,
        'all_data'     : bool(args.all_data),
        'data_splitted': bool(args.data_splitted),
        'to_shuffle'    : bool(args.to_shuffle)
    }

    return dict_args

# Generator
def create_generator(latent_size):
    gen = Sequential()
    gen.add(Dense(latent_size, input_dim=latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
    gen.add(Dense(latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
    return gen

# Discriminator
def create_discriminator():
    dis = Sequential()
    dis.add(Dense(math.ceil(math.sqrt(data_size)), input_dim=latent_size, activation='relu', 
            kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', 
            distribution='normal', 
            seed=None)))
    dis.add(Dense(math.ceil(math.sqrt(math.sqrt(data_size))), activation='relu', 
            kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', 
            distribution='normal', 
            seed=None)))
    dis.add(Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    #dis.compile(optimizer=SGD(learning_rate=args["lr"], decay=args["decay"], momentum=args["momentum"]), loss='binary_crossentropy')
    return dis

# Load data
def load_data(path_data, all_data):

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

def load_data_splitted(path_data, to_shuffle):
    path_data_train, path_data_test = path_data 
    df_train = pd.read_csv(path_data_train, header=None).sample(frac=1).reset_index(drop=True)
    df_test = pd.read_csv(path_data_test, header=None).sample(frac=1).reset_index(drop=True)

    label_train = pd.read_csv(path_data_train.replace(".csv", "") + "_label.csv", header=None)
    label_test = pd.read_csv(path_data_test.replace(".csv", "") + "_label.csv", header=None)

    

    if to_shuffle:
        column_name = "label"
        label_train.columns = [column_name]
        label_test.columns = [column_name]

        df_train = pd.concat([df_train, label_train], axis=1).sample(frac=1).reset_index(drop=True)
        df_test = pd.concat([df_test, label_test], axis=1).sample(frac=1).reset_index(drop=True)

        label_train = df_train.pop(column_name)
        label_test = df_test.pop(column_name)

    print("size data")
    print("train data")
    print(df_train.values.shape)
    print("train label")
    print(label_train.values.shape)
    print("test data")
    print(df_test.values.shape)
    print("test label")
    print(label_test.values.shape)

    assert(df_train.values.shape[0] == label_train.values.shape[0])
    assert(df_test.values.shape[0] == label_test.values.shape[0])
    assert(df_train.values.shape[1] == df_test.values.shape[1])

    return df_train.values, label_train.values, df_test.values, label_test.values

# Plot loss history
def plot(train_history, title, field, args, label):
    #dy = train_history['discriminator_loss']
    #gy = train_history['generator_loss']
    #x = np.linspace(1, len(aucy), len(aucy))
    #ax.plot(x, dy, color='green')
    #ax.plot(x, gy,color='red')
    
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

    name = "res/vect_res_{}_db_{}_LR_{}_momentum_{}_decay_{}_{}_{}.png".format(field, args["path"].replace("/", "-"), args["lr"], args["momentum"], args["decay"], date.today(), current_time.replace(":", "-"))

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

def calc_auc(train_history, field, to_print, discriminator, data_x, data_y, other):
    acc = 0
    if not other:
        # Detection result
        p_value = discriminator.predict(data_x)
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
        acc = (sum / (len(inlier_parray) * len(outlier_parray)))

    else:
        pred = discriminator.predict(data_x)
        _, _, thresholds = metrics.roc_curve(data_y, pred)

        
        for thres in thresholds:
            y_pred = np.where(pred < thres, 1, 0)
            acc_tmp = metrics.accuracy_score(data_y, y_pred)
            if acc_tmp > acc:
                acc = acc_tmp
    
    print(to_print + "  " +"{:.4f}".format(acc))
    train_history[field].append(acc)


if __name__ == '__main__':
    train = True
    args = parse_args()
    if not args["data_splitted"]:
        data_x, data_y = load_data(args["path"], args["all_data"]) # faut mettre le dossier, apres load_data se charge du reste
        rows = np.random.choice(data_x.shape[0], size=data_x.shape[0] // 10, replace=True)
        data_x_test = data_x[rows]
        data_x = data_x[~rows]
        data_y_test = data_y[rows]
        data_y = data_y[~rows]

    else:
        data_x, data_y, data_x_test, data_y_test = load_data_splitted(args["path"].split("%"), args['to_shuffle'])
    print("The dimension of the training data :{}*{}".format(data_x.shape[0], data_x.shape[1]))
    if train:
        latent_size = data_x.shape[1]
        data_size = data_x.shape[0]
        stop = 0
        epochs = args["stop_epochs"] * 3
        train_history = defaultdict(list)

        # Create discriminator
        discriminator = create_discriminator()
        discriminator.compile(optimizer=SGD(learning_rate=args["lr"], decay=args["decay"], momentum=args["momentum"]), loss='binary_crossentropy')

        # Create combine model
        generator = create_generator(latent_size)
        latent = Input(shape=(latent_size,))
        fake = generator(latent)
        discriminator.trainable = False
        fake = discriminator(fake)
        combine_model = Sequential()
        combine_model.add(generator)
        combine_model.add(discriminator)
        combine_model.compile(optimizer=SGD(learning_rate=args["lr"], decay=args["decay"], momentum=args["momentum"]), loss='binary_crossentropy')

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
                generated_data = generator.predict(noise, verbose=0)

                # Concatenate real data to generated data
                X = np.concatenate((data_batch, generated_data))
                Y = np.array([1] * batch_size + [0] * int(noise_size))

                # Train discriminator
                discriminator_loss = discriminator.train_on_batch(X, Y)
                train_history['discriminator_loss'].append(discriminator_loss)

                # Train generator
                if stop == 0:
                    trick = np.array([1] * noise_size)
                    generator_loss = combine_model.train_on_batch(noise, trick)
                    train_history['generator_loss'].append(generator_loss)
                else:
                    trick = np.array([1] * noise_size)
                    generator_loss = combine_model.evaluate(noise, trick)
                    train_history['generator_loss'].append(generator_loss)

            # Stop training generator
            if epoch + 1 > args["stop_epochs"]:
                stop = 1

            calc_auc(train_history, 'auc', "AUC", discriminator, data_x, data_y, args['auc_other'])

            calc_auc(train_history, 'auc_test', "AUC_test", discriminator, data_x_test, data_y_test, args['auc_other'])
        
        print("maintenant on affiche")
        plot(train_history, 'ROC AUC', 'auc', args, 'train acc')
        plot(train_history, 'discriminator_loss', 'discriminator_loss', args, 'discri loss')
        plot(train_history, 'generator_loss', 'generator_loss', args, 'gene loss')
