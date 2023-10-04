import hydra
import ray
import os


ray.init(address="auto")


@hydra.main(config_path="../conf", config_name="default", version_base="1.1")
def my_app(cfg):
    hydra.utils.call(cfg.task, _recursive_=True)

if __name__ == "__main__":
    my_app()