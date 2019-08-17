import time
import parameters
from train import trainer
def main():
    args = parameters.get_parser()
    print(args.config)
    Experiment=trainer(args=args,time_stp=time_stp)
    Experiment.train()

if __name__ == '__main__':
    time_stp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    main()