import hydra
import ray


ray.init(address="auto")

tasks = []

@hydra.main(config_path="../conf", config_name="default", version_base="1.1")
def my_app(cfg):
    tasks.append(hydra.utils.call(cfg.task, _recursive_=True))

if __name__ == "__main__":
    my_app()
    ray.get(tasks)
