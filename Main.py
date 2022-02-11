from Experiments import Experiments
from models.Vanilla import Vanilla


config_file = 'config.ini'
n_steps = 4
n_seq = 2
n_features = 1

exp = Experiments(config_file=config_file)

model = Vanilla(n_steps=n_steps, n_features=n_features, learning_rate=1e-3, loss='mse', metrics=['mape', 'mse', 'mae'])
exp.set_model(model)
exp.set_monitoring_time(60)
# set time train
# set time eval
exp.set_monitoring_sleep(1)
exp.run_monitoring()

exp.run_model(metric_name='MemUsed',
              train_size=0.5,
              epochs=30,
              n_steps=n_steps,
              n_features=n_features,
              y_step=10,
              reshape='linear')

# exp.eval_best_model(metric_name='MemUsed',
#                     model_metrics=['mape', 'mse', 'mae'],
#                     eval_metric='mse',
#                     learning_rate=1e-3,
#                     loss='mse',
#                     train_size=0.5,
#                     epochs=1,
#                     n_steps=n_steps,
#                     n_features=n_features,
#                     n_seq=n_seq)