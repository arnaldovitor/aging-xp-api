import configparser
import os
import subprocess
import time
import random

from models.Vanilla import Vanilla

import matplotlib.pyplot as plt
import Utils as u
import numpy as np


class Experiments:
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        self.metrics = ['MemUsed', 'MemShared', 'MemBuffCache', 'DiskUsed', 'DiskPerc', '1kBlocks', 'UsrPerc', 'SysPerc', 'IoWaitPerc', 'SoftPerc', 'IdlePerc']
        self.time = None
        self.sleep = None
        self.model = None

    def set_time(self, time):
        self.time = time

    def set_sleep(self, sleep):
        self.sleep = sleep

    def set_model(self, model):
        self.model = model

    def __countdown(self, t):
        while t:
            m, s = divmod(t, 60)
            min_sec_format = '{:02d}:{:02d}'.format(m, s)
            print(min_sec_format, end='\n')
            time.sleep(1)
            t -= 1

        print('Countdown finished.')

    def __gen_log(self):
        log = open(os.path.join(self.config['PATHS']['log'], 'script.sh'), 'w')
        log.write('echo {} > {}\n\n'.format(' '.join(self.metrics), os.path.join(self.config['PATHS']['log'], 'log.txt')))
        log.write('while [ TRUE ]\n')
        log.write('do\n\n')
        log.write("mem=`free | grep Mem | awk '{print $3, $5, $6}'`\n")
        log.write("disco=`df | grep sda2 | awk '{print $3, $5, $2}'`\n")
        log.write("cpu=`mpstat | grep all | awk '{print $3, $4, $5, $7, $11}'`\n\n")
        log.write('echo $mem $disco $cpu >> {}\n\n'.format(os.path.join(self.config['PATHS']['log'], 'log.txt')))
        log.write('sleep {}\n\n'.format(self.sleep))
        log.write('done')
        log.close()

    def run_exp(self):
        self.__gen_log()
        os.popen('chmod +x -R {}'.format(self.config['PATHS']['log']))
        p = subprocess.Popen('exec {}'.format(os.path.join(self.config['PATHS']['log'], 'script.sh')),  stdout=subprocess.PIPE, shell=True)
        self.__countdown(self.time)
        p.kill()

    def run_model(self, metric_name, train_size, epochs, n_steps, n_features, normalize=True):
        sequence = u.separe_column(r"{}".format(os.path.join(self.config['PATHS']['log'], 'log.txt')), metric_name)

        if normalize:
            sequence, s_min, s_max = u.normalize(sequence)

        train, test = u.split_sets(sequence, train_size)

        X, y = u.split_sequence(train.tolist(), n_steps)
        X_test, y_test = u.split_sequence(test.tolist(), n_steps)

        X = X.astype(np.float)
        y = y.astype(np.float)
        X_test = X_test.astype(np.float)
        y_test = y_test.astype(np.float)

        X = X.reshape((X.shape[0], X.shape[1], n_features))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

        self.model.fit(X, y, validation_data=(X_test, y_test), epochs=epochs, verbose=1)

        pred_x = self.model.predict(X)
        pred_x_test = self.model.predict(X_test)

        dif = (len(np.concatenate((y, y_test), axis=0)) - len(pred_x_test))

        axis_x_test = [(i + dif) for i in range(len(pred_x_test))]
        y_true = np.concatenate((y, y_test), axis=0)

        plt.plot(y_true, label='Original Set', color='blue')
        plt.plot(pred_x, label='Predicted Train Set', color='red', linestyle='-.')
        plt.plot(axis_x_test, pred_x_test, label='Predicted Test Set', color='green', linestyle='-.')

        plt.xlabel('Time (sec)')
        plt.ylabel(metric_name)
        plt.legend()

        hash = str(random.getrandbits(128))
        plt.savefig(os.path.join(self.config['PATHS']['plots'], "{}.png".format(hash)), dpi=800)

        plt.show()

if __name__ == '__main__':
    config_file = 'config.ini'
    n_steps = 4
    n_features = 1

    exp = Experiments(config_file=config_file)
    model = Vanilla(n_steps=n_steps, n_features=n_features, learning_rate=1e-3, loss='mse', metrics=['mape', 'mse', 'mae'])
    exp.set_model(model)
    exp.set_time(50)
    exp.set_sleep(1)
    exp.run_exp()
    exp.run_model(metric_name='MemUsed', train_size=0.8, epochs=2, n_steps=n_steps, n_features=n_features)
