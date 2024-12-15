import os
import logging
from src.utils.gpu_config import configure_gpu
from src.data.loader import DataLoader
from src.models.mobilenet import MobileNetV2Model
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model
from configs import config

def main():
    # Step 1: Configure GPU
    configure_gpu(config['gpu_memory_limit'])

    # Step 2: Initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting the FER Emotion Detection pipeline...")

    # Step 3: Data Preparation
    logger.info("Loading and preprocessing data...")
    data_loader = DataLoader(config['data']['path'], config['data']['batch_size'], config['data']['augment'])
    train_data, val_data, test_data = data_loader.get_data_splits()

    # Step 4: Model Initialization
    logger.info("Initializing the MobileNetV2 model...")
    model = MobileNetV2Model(input_shape=config['model']['input_shape'], num_classes=config['model']['num_classes'])
    model.build_model()

    # Step 5: Training
    logger.info("Starting training...")
    trainer = Trainer(model=model, train_data=train_data, val_data=val_data, config=config['training'])
    trainer.train()

    # Step 6: Evaluation
    logger.info("Evaluating the model...")
    metrics = evaluate_model(model, test_data)
    logger.info(f"Evaluation Metrics: {metrics}")

    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
