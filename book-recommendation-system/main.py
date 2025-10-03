import argparse
import logging
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# TODO: Import modules when they are created
# from src.data.make_dataset import main as process_data
# from src.models.train_model import main as train_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Book Recommendation System Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    subparsers.add_parser('process', help='Process raw data.')
    subparsers.add_parser('train', help='Train the models.')

    api_parser = subparsers.add_parser('api', help='Run the FastAPI server.')
    api_parser.add_argument('--host', type=str, default='127.0.0.1')
    api_parser.add_argument('--port', type=int, default=8000)
    api_parser.add_argument('--reload', action='store_true')

    args = parser.parse_args()

    if args.command == 'process':
        logger.info("Running data processing...")
        # process_data()
    elif args.command == 'train':
        logger.info("Running model training...")
        # train_models()
    elif args.command == 'api':
        logger.info(f"Starting API server on {args.host}:{args.port}...")
        try:
            import uvicorn
            uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=args.reload)
        except ImportError:
            logger.error("uvicorn not found. Install with 'pip install uvicorn'.")
    else:
        parser.print_help()

if __name__ == "__main__":
    if not os.path.exists('logs'):
        os.makedirs('logs')
    main()
