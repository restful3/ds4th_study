import argparse
from config import config, update_config
from stocktrainer import StockTrainer
from stockevaluator import StockEvaluator
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Stock Trading RL')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'both'], default='both', help='Operation mode')
    parser.add_argument('--agent', choices=['dqn', 'ppo', 'a2c'], default='dqn', help='RL agent type')
    parser.add_argument('--ticker', type=str, default='005930', help='Stock ticker')
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_arguments()
    
    update_config({'agent_type': args.agent, 'ticker': args.ticker})
    
    if args.mode in ['train', 'both']:
        trainer = StockTrainer()
        trainer.train()
        trainer.save_model()
        
    if args.mode in ['evaluate', 'both']:
        evaluator = StockEvaluator(args.ticker)
        evaluator.run_evaluation()
    
    logging.info("Process completed successfully.")

if __name__ == "__main__":
    main()