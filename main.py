import mrtm
import elastictransformer as et
import torch

params = et.ParameterProvider("series-weather.config")

dataset = et.CustomDataSet('dane_pogodowe.csv',window_length=720,prediction_window=30)

device_id = torch.cuda.current_device()


runner = mrtm.MRTM(et.ElasticTransformer(config=params).cuda(device_id),dump_name="dumps",dump_folder="$SCRATCH/dumps")

train_dataset, test_dataset = dataset.getSets()

runner.run(1,5,torch.cuda.current_device(),train_dataset,25,train_dataset,0.15,0.1,0.35,2.0,4,4,backup_delay=1)

runner.generate_report(train_dataset,device_id,path="$SCRATCH/dumps")