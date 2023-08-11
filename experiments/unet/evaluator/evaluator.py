import torch
import wandb
import tqdm
import time

from torch.utils.data import DataLoader
from monai.losses import DiceLoss

from unet.dataset.msd_dataset import Split, MSDTask, MSDDataset

class Evaluator():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_TYPE = 'unet'
    PROJECT_NAME = 'UNet'
    
    def __init__(self, config):
        assert config['model_type'] == self.MODEL_TYPE
        torch.set_float32_matmul_precision('medium')
        self.run_name = config['name']
        self.data_loading_config = config['data_loading']
        self.model_loading_config = config['model_loading']
        self.logging_config = config['logging']
        self.eval_config = config['eval']
        
        if self.logging_config['enabled'] == True:
            wandb.init(project=self.PROJECT_NAME, name=self.run_name, config=config)
        
        self.task = MSDTask.fromStr(self.data_loading_config['task'])
        split_ratio = (
            self.data_loading_config['split_ratio']['train'],
            self.data_loading_config['split_ratio']['val'],
            self.data_loading_config['split_ratio']['test']
        )
        self.test_dataset = MSDDataset(
            msd_task=self.task,
            split=Split.Test,
            split_ratio=split_ratio,
            seed=self.data_loading_config['seed'],
            normalize=True
        )
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.data_loading_config['batch_size'],
            num_workers=self.data_loading_config['n_workers'],
            shuffle=False,
        )
        self.initModel()
        self.loss_fn = DiceLoss(softmax=True, include_background=False)
        self.epochs = self.eval_config['n_steps'] // (len(self.test_dataset) // self.data_loading_config['batch_size'])

    def initModel(self):
        raise NotImplementedError
    
    def infer(self, scan):
        raise NotImplementedError
    
    def evaluate(self):
        loss_list = []
        gpu_mem_list = []
        gpu_cache_list = []
        time_list = []
        for epoch in range(self.epochs):
            loop = tqdm.tqdm(
                self.test_loader,
                desc=f"Test-Epoch {epoch + 1}/{self.epochs}"
            )
            for batch_idx, (data, targets) in enumerate(loop):
                data = data.to(self.DEVICE)
                start_time = time.time()
                with torch.cuda.amp.autocast():
                    predictions = self.infer(data)
                    end_time = time.time()
                    gpu_mem = torch.cuda.memory_allocated(self.DEVICE)
                    gpu_cache = torch.cuda.memory_reserved(self.DEVICE)
                    loss = self.loss_fn(predictions, targets.to(self.DEVICE))
                loss_list.append(loss.item())
                gpu_mem_list.append(gpu_mem)
                gpu_cache_list.append(gpu_cache)
                time_list.append(end_time - start_time)
                if self.logging_config['enabled'] == True:
                    wandb.log({"test_loss": loss.item()})
                    wandb.log({"test_gpu_mem": gpu_mem})
                    wandb.log({"test_gpu_cache": gpu_cache})
                    wandb.log({"test_time": end_time - start_time})
        avrg_loss = sum(loss_list) / len(loss_list)
        avrg_gpu_mem = sum(gpu_mem_list) / len(gpu_mem_list)
        avrg_gpu_cache = sum(gpu_cache_list) / len(gpu_cache_list)
        avrg_time = sum(time_list) / len(time_list)
        if self.logging_config['enabled'] == True:
            wandb.log({"test_avrg_loss": avrg_loss})
            wandb.log({"test_avrg_gpu_mem": avrg_gpu_mem})
            wandb.log({"test_avrg_gpu_cache": avrg_gpu_cache})
            wandb.log({"test_avrg_time": avrg_time})
