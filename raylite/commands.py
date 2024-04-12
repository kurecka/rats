from functools import partial
import os
import multiprocessing
from termcolor import cprint
from omegaconf import DictConfig


def run_command(cmd, dry_run):
    cprint('  ' + ('[DRY RUN] ' if dry_run else '') + 'Running bash command:', 'blue')
    print('  ' + cmd)
    if not dry_run:
        os.system(cmd)



def run_remote_command(host, user, cmd, container_name, dry_run):
    if container_name is None:
        command = f'ssh {user}@{host} "{cmd}"'
    else:
        command = f'ssh {user}@{host} "docker exec {container_name} {cmd}"'

    run_command(cmd=command, dry_run=dry_run)

def run_remote_root(host, user, cmd, container_name, dry_run):
    assert( container_name )
    command = f'ssh {user}@{host} "docker exec --user root {container_name} {cmd}"'
    run_command(cmd=command, dry_run=dry_run)



def run_remote_commands(host, user, cmds, container_name, dry_run):
    for cmd in cmds:
        run_remote_command(host=host, user=user, cmd=cmd, container_name=container_name, dry_run=dry_run)


def ray_cluster(provider,
                user,
                head_cmds: list[str],
                worker_cmds: list[str],
                action_msg: str,
                dry_run: bool,
                container_name: str | None) -> None:
    cprint(f'{action_msg} head node {provider.head_ip}...', 'green')
    run_remote_commands(
        host=provider.head_ip,
        user=user,
        cmds=head_cmds,
        container_name=container_name,
        dry_run=dry_run
    )

    cprint(f'{action_msg} worker nodes {provider.worker_ips}...', 'green')
    run_remote = partial(
        run_remote_commands,
        user=user,
        cmds=worker_cmds,
        container_name=container_name,
        dry_run=dry_run
    )
    with multiprocessing.Pool(processes=max(len(provider.worker_ips), 1)) as pool:
        pool.map(run_remote, provider.worker_ips)


def ray_up(cfg: DictConfig, dry_run: bool) -> None:
    head_cmds = cfg.head_start_ray_commands
    worker_cmds = cfg.worker_start_ray_commands
    if cfg.docker is not None:
        cmds = [f'docker stop {cfg.docker.container_name}']
        if cfg.docker.pull_before_run:
            cmds.append(f'docker pull {cfg.docker.image}')
        docker_cmd = (f'docker run --network host -d -it --rm --name {cfg.docker.container_name}'
                      f' {" ".join(cfg.docker.run_options)} {cfg.docker.image}')
        
        head_cmd = docker_cmd + f" sh -c '{' && '.join(head_cmds)}; bash'"
        head_cmds = cmds + [head_cmd]

        worker_cmd = docker_cmd + f" sh -c '{' && '.join(worker_cmds)}; bash'"
        worker_cmds = cmds + [worker_cmd]


    ray_cluster(
        provider=cfg.provider,
        user=cfg.auth.ssh_user,
        head_cmds=head_cmds,
        worker_cmds=worker_cmds,
        action_msg='Starting',
        container_name=None,
        dry_run=dry_run
    )

    chown_rats(cfg=cfg, dry_run=dry_run)

def chown_rats(cfg, dry_run):
    for worker in cfg.provider.worker_ips:
        for cmd in cfg.worker_start_privileged_commands:
            print("Running " + cmd + " privileged on worker " + worker)
            run_remote_root(worker, cfg.auth.ssh_user, cmd, cfg.docker.container_name, dry_run)

def ray_down(cfg: DictConfig, dry_run: bool) -> None:
    head_cmds = cfg.head_stop_ray_commands
    worker_cmds = cfg.worker_stop_ray_commands
    if cfg.docker is not None:
        head_cmds = [f'docker stop {cfg.docker.container_name}']
        worker_cmds = [f'docker stop {cfg.docker.container_name}']

    ray_cluster(
        provider=cfg.provider,
        user=cfg.auth.ssh_user,
        head_cmds=head_cmds,
        worker_cmds=worker_cmds,
        action_msg='Stopping',
        container_name=None,
        dry_run=dry_run
    )


def ray_dashboard(cfg: DictConfig, dry_run: bool) -> None:
    cprint(f'Forwarding dashboard port {cfg.dashboard_port} to'
           f' http://localhost:{cfg.dashboard_port}...', 'green')
    run_command(cmd=cfg.dashboard_command, dry_run=dry_run)


def ray_attach(cfg: DictConfig, dry_run: bool) -> None:
    run_command(cmd=cfg.attach_command, dry_run=dry_run)


def git_pull(cfg: DictConfig, dry_run: bool) -> None:
    ray_cluster(
        provider=cfg.provider,
        user=cfg.auth.ssh_user,
        head_cmds=['git pull'],
        worker_cmds=['git pull'],
        action_msg='Pulling',
        container_name=cfg.docker.container_name,
        dry_run=dry_run
    )
