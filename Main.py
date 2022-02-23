from Experiments import Experiments
from models.Vanilla import Vanilla


config_file = 'config.ini'
n_steps = 4
n_seq = 2
n_features = 1
y_step = 5

exp = Experiments(config_file=config_file)

model = Vanilla(n_steps=n_steps, n_features=n_features, learning_rate=0.5e-3, loss='mse', metrics=['mape', 'mse', 'mae'])

exp.set_monitoring_time(60*60)
exp.set_monitoring_sleep(1)
exp.run_monitoring()

exp.run_model(model=model,
              metric_name='MemUsed',
              train_size=0.8,
              epochs=800,
              n_steps=n_steps,
              n_features=n_features,
              y_step=y_step,
              reshape='linear')

exp.add_model(hash='69588700973967516361194600616928228529',
              threshold=300,
              metric='MemUsed',
              reshape='linear')

exp.set_forecast_time(60*60)
exp.set_forecast_sleep(5)
exp.run_forecast(rejuvenation_script='',
                 n_steps=n_steps,
                 n_features=n_features,
                 y_step=y_step)

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