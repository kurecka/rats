#!/usr/bin/env python
import click
from omegaconf import OmegaConf
from raylite.commands import (
    ray_up,
    ray_down,
    ray_dashboard,
    ray_attach,
    git_pull,
    chown_rats,
    sync_head_to_workers,
    head_cmd,
    workers_cmd,
)
from raylite.utils import set_user
import os


OmegaConf.register_new_resolver("user", lambda: os.getlogin())
OmegaConf.register_new_resolver("uid", lambda: os.getuid())
OmegaConf.register_new_resolver("gid", lambda: os.getgid())


@click.command()
@click.argument('action', type=click.Choice(['up', 'down', 'dashboard', 'attach', 'git-pull', 'chown', 'sync', 'hcmd', 'wcmd']))
@click.argument('config', type=click.Path(exists=True), default='ray_launch.yaml')

@click.option('--cmd', type=str, default=None, help='Command to run on the head or worker nodes.')
@click.option('--user', '-u', type=str, default=None,
              help='SSH user name. Overwrite `auth.ssh_user` in the config file.')
@click.option('--dry-run', '-d', is_flag=True,
              help='Dry run.')
@click.option('--print-config', '-p', is_flag=True, default=False,
              help='Prints the config file and exits.')
@click.help_option('-h', '--help')
def main(action, config, user, dry_run, print_config, cmd):
    """A lightweight replacement of a malfunction ray cluster commands."""
    cfg = OmegaConf.load(config)

    set_user(conf=cfg, user=user)

    if print_config:
        print(OmegaConf.to_yaml(cfg))
        exit(0)

    if action == 'up':
        ray_up(cfg=cfg, dry_run=dry_run)

    elif action == 'chown':
        chown_rats(cfg=cfg, dry_run=dry_run)

    elif action == 'down':
        ray_down(cfg=cfg, dry_run=dry_run)

    elif action == 'dashboard':
        ray_dashboard(cfg=cfg, dry_run=dry_run)

    elif action == 'attach':
        ray_attach(cfg=cfg, dry_run=dry_run)

    elif action == 'git-pull':
        git_pull(cfg=cfg, dry_run=dry_run)
    
    elif action == 'sync':
        sync_head_to_workers(cfg=cfg, dry_run=dry_run)

    elif action == 'hcmd':
        assert cmd is not None, "Please provide a command to run on the head node."
        head_cmd(cfg=cfg, cmd=cmd, dry_run=dry_run)
        
    elif action == 'wcmd':
        assert cmd is not None, "Please provide a command to run on the worker nodes."
        workers_cmd(cfg=cfg, cmd=cmd, dry_run=dry_run)
    
    else:
        raise NotImplementedError(f"Action {action} is not implemented.")


if __name__ == '__main__':
    main()
