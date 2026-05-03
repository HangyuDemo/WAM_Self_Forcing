import hydra
from omegaconf import DictConfig

from scripts.train_idm_self_forcing import run_self_forcing_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train_idm_self_forcing_proprio_joint", version_base="1.3")
def main(cfg: DictConfig):
    run_self_forcing_training(cfg)


if __name__ == "__main__":
    main()
