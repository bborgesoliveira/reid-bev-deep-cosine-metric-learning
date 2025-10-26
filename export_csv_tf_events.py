import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# Caminho para o arquivo de logs
logdir = "./runs/dual_loss_64channels"

# Carregar os eventos
event_acc = event_accumulator.EventAccumulator(logdir)
event_acc.Reload()

# Lista todas as tags disponíveis no TensorBoard
tags = event_acc.Tags()['scalars']

# Exporta os dados para CSV
for tag in tags:
    events = event_acc.Scalars(tag)
    df = pd.DataFrame(events)
    df.columns = ['wall_time', 'step', 'value']
    df.to_csv(f"{tag.replace('/', '_')}.csv", index=False)

print("Exportação concluída!")