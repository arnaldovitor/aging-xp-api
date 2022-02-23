import configparser
import os
import random
import subprocess
import time
import sys

from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
import Utils as u

from models.Bidirectional import Bidirectional
from models.CNN import CNN
from models.Conv import Conv
from models.Stacked import Stacked
from models.Vanilla import Vanilla


class Experiments:
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.metrics = ['MemUsed', 'MemShared', 'MemBuffCache', 'DiskUsed', 'DiskPerc', '1kBlocks', 'UsrPerc', 'SysPerc', 'IoWaitPerc', 'SoftPerc', 'IdlePerc']
        self.monitoring_time = None
        self.monitoring_sleep = None
        self.forecast_time = None
        self.forecast_sleep = None
        self.model_list = []
        self.hash_list = []
        self.threshold_list = []
        self.metrics_list = []
        self.reshape_list = []
        self.min_list = []
        self.max_list = []

    def set_monitoring_time(self, monitoring_time):
        self.monitoring_time = monitoring_time

    def set_monitoring_sleep(self, monitoring_sleep):
        self.monitoring_sleep = monitoring_sleep

    def set_forecast_time(self, forecast_time):
        self.forecast_time = forecast_time

    def set_forecast_sleep(self, forecast_sleep):
        self.forecast_sleep = forecast_sleep

    def __print_progbar(self, i, max, text):
        n_bar = 40
        j = i / max
        sys.stdout.write('\r')
        sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {text}")
        sys.stdout.flush()

    def __countdown(self, t, progbar=False):
        for i in range(t):
            if progbar == True:
                self.__print_progbar(i, t, 'Progress')
            time.sleep(1)

    def __gen_monitoring_script(self):
        log = open(os.path.join(self.config['PATHS']['scripts'], 'monitoring_script.sh'), 'w')
        log.write('echo {} > {}\n\n'.format(' '.join(self.metrics), os.path.join(self.config['PATHS']['logs'], 'monitoring_log.txt')))
        log.write('while [ TRUE ]\n')
        log.write('do\n\n')
        log.write("mem=`free | grep Mem | awk '{print $3, $5, $6}'`\n")
        log.write("disco=`df | grep sda2 | awk '{print $3, $5, $2}'`\n")
        log.write("cpu=`mpstat | grep all | awk '{print $3, $4, $5, $7, $11}'`\n\n")
        log.write('echo $mem $disco $cpu >> {}\n\n'.format(os.path.join(self.config['PATHS']['logs'], 'monitoring_log.txt')))
        log.write('sleep {}\n\n'.format(self.monitoring_sleep))
        log.write('done')
        log.close()

    def __gen_forecast_script(self):
        log = open(os.path.join(self.config['PATHS']['scripts'], 'forecast_script.sh'), 'w')
        log.write('echo {} > {}\n\n'.format(' '.join(self.metrics), os.path.join(self.config['PATHS']['logs'], 'forecast_log.txt')))
        log.write('while [ TRUE ]\n')
        log.write('do\n\n')
        log.write("mem=`free | grep Mem | awk '{print $3, $5, $6}'`\n")
        log.write("disco=`df | grep sda2 | awk '{print $3, $5, $2}'`\n")
        log.write("cpu=`mpstat | grep all | awk '{print $3, $4, $5, $7, $11}'`\n\n")
        log.write('echo $mem $disco $cpu >> {}\n\n'.format(os.path.join(self.config['PATHS']['logs'], 'forecast_log.txt')))
        log.write('done')
        log.close()

    def __save_model(self, model, hash):
        model.save(os.path.join(self.config['PATHS']['models'], '{}'.format(hash)))

    def run_monitoring(self, monitoring_script='monitoring_script.sh'):
        if monitoring_script == 'monitoring_script.sh':
            self.__gen_monitoring_script()
        os.popen('chmod +x -R {}'.format(self.config['PATHS']['scripts']))
        p = subprocess.Popen('exec {}'.format(os.path.join(self.config['PATHS']['scripts'], monitoring_script)), stdout=subprocess.PIPE, shell=True)
        print('# MONITORING')
        self.__countdown(self.monitoring_time, True)
        p.kill()

    def run_forecast(self, rejuvenation_script,n_steps, n_features, y_step, n_seq=None, normalize=True, forecast_log='forecast_log.txt', forecast_script='forecast_script.sh'):
        if forecast_script == 'forecast_script.sh':
            self.__gen_forecast_script()
        os.popen('chmod +x -R {}'.format(self.config['PATHS']['scripts']))

        forecast_start = time.time()
        t = 0

        print('# FORECAST')
        while time.time() < forecast_start + self.forecast_time:
            for i, model in enumerate(self.model_list):
                flag_list = []
                p1 = subprocess.Popen('exec {}'.format(os.path.join(self.config['PATHS']['scripts'], forecast_script)), stdout=subprocess.PIPE, shell=True)
                self.__countdown(self.forecast_sleep)
                p1.kill()

                if normalize:
                    sequence, _, _ = u.normalize(sequence, self.min_list[i], self.max_list[i])

                sequence = u.separe_column(r"{}".format(os.path.join(self.config['PATHS']['logs'], forecast_log)), self.metrics_list[i])
                sequence, _ = u.split_sequence(sequence.tolist(), n_steps, y_step)
                sequence.astype(np.float)

                if self.reshape_list[i] == 'linear':
                    sequence = sequence.reshape((sequence.shape[0], sequence.shape[1], n_features))
                elif self.reshape_list[i] == 'cnn':
                    n_steps = int(n_steps / n_seq)
                    sequence = sequence.reshape((sequence.shape[0], n_seq, n_steps, n_features))
                elif self.reshape_list[i] == 'conv':
                    n_steps = int(n_steps / n_seq)
                    sequence = sequence.reshape((sequence.shape[0], n_seq, 1, n_steps, n_features))

                pred = model.predict(sequence)

                if float(pred[0]) > self.threshold_list[i]:
                    flag_list.append(1)
                else:
                    flag_list.append(0)

            if flag_list.count(1) >= flag_list.count(0):
                p2 = subprocess.Popen('exec {}'.format(os.path.join(self.config['PATHS']['scripts'], rejuvenation_script)), stdout=subprocess.PIPE, shell=True)
                p2.kill()
                print('\n# ACTIVATED REJUVENATION')
                print('Flag list:', flag_list)

            t = time.time() - forecast_start
            self.__print_progbar(t, self.forecast_time, 'Progress')


    def run_model(self, model, metric_name, train_size, epochs, n_steps, n_features, y_step, reshape, log='monitoring_log.txt', n_seq=None, normalize=True):
        sequence = u.separe_column(r"{}".format(os.path.join(self.config['PATHS']['logs'], log)), metric_name)

        if normalize:
            sequence, s_min, s_max = u.normalize(sequence)

        train, test = u.split_sets(sequence, train_size)

        X, y = u.split_sequence(train.tolist(), n_steps, y_step)
        X_test, y_test = u.split_sequence(test.tolist(), n_steps, y_step)

        X = X.astype(np.float)
        y = y.astype(np.float)
        X_test = X_test.astype(np.float)
        y_test = y_test.astype(np.float)

        if reshape == 'linear':
            X = X.reshape((X.shape[0], X.shape[1], n_features))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
        elif reshape == 'cnn':
            n_steps = int(n_steps / n_seq)
            X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
            X_test = X_test.reshape((X_test.shape[0], n_seq, n_steps, n_features))
        elif reshape == 'conv':
            n_steps = int(n_steps / n_seq)
            X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
            X_test = X_test.reshape((X_test.shape[0], n_seq, 1, n_steps, n_features))


        history = model.fit(X, y, validation_data=(X_test, y_test), epochs=epochs, verbose=1)

        pred_x = model.predict(X)
        pred_x_test = model.predict(X_test)

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

        self.__save_model(model, hash)

        if normalize:
            return history, s_min, s_max
        else:
            return history

    def add_model(self, hash, threshold, metric, reshape, s_min, s_max):
        model = tf.keras.models.load_model(os.path.join(self.config['PATHS']['models'], hash))
        self.model_list.append(model)
        self.hash_list.append(hash)
        self.threshold_list.append(threshold)
        self.metrics_list.append(metric)
        self.reshape_list.append(reshape)
        self.min_list.append(s_min)
        self.max_list.append(s_max)


    def eval_best_model(self, metric_name, model_metrics, eval_metric, learning_rate, loss, train_size, epochs, n_steps, n_features, n_seq):
        eval = {'Vanilla': 0, 'Bidirectional': 0, 'Stacked': 0, 'CNN': 0, 'Conv': 0}

        model = Vanilla(n_steps=n_steps, n_features=n_features, learning_rate=learning_rate, loss=loss, metrics=model_metrics)
        self.set_model(model)
        history = self.run_model(metric_name=metric_name, train_size=train_size, epochs=epochs, n_steps=n_steps, n_features=n_features, reshape='linear')
        eval['Vanilla'] = history.history[eval_metric][0]

        model = Bidirectional(n_steps=n_steps, n_features=n_features, learning_rate=learning_rate, loss=loss, metrics=model_metrics)
        self.set_model(model)
        history = self.run_model(metric_name=metric_name, train_size=train_size, epochs=epochs, n_steps=n_steps, n_features=n_features, reshape='linear')
        eval['Bidirectional'] = history.history[eval_metric][0]

        model = Stacked(n_steps=n_steps, n_features=n_features, learning_rate=learning_rate, loss=loss, metrics=model_metrics)
        self.set_model(model)
        history = self.run_model(metric_name=metric_name, train_size=train_size, epochs=epochs, n_steps=n_steps, n_features=n_features, reshape='linear')
        eval['Stacked'] = history.history[eval_metric][0]

        model = CNN(n_steps=int(n_steps / n_seq), n_features=n_features, learning_rate=learning_rate, loss=loss, metrics=model_metrics)
        self.set_model(model)
        history = self.run_model(metric_name=metric_name, train_size=train_size, epochs=epochs, n_steps=n_steps, n_features=n_features, reshape='cnn', n_seq=n_seq)
        eval['CNN'] = history.history[eval_metric][0]

        model = Conv(n_steps=int(n_steps / n_seq), n_features=n_features, learning_rate=learning_rate, loss=loss, metrics=model_metrics, n_seq=n_seq)
        self.set_model(model)
        history = self.run_model(metric_name=metric_name, train_size=train_size, epochs=epochs, n_steps=n_steps, n_features=n_features, reshape='conv', n_seq=n_seq)
        eval['Conv'] = history.history[eval_metric][0]

        eval_ord = sorted(eval.items(), key=lambda item: item[1])
        print('Models in ascending order using the {} metric'.format(eval_metric))
        print(eval_ord)
