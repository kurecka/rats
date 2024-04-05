import getpass
import omegaconf


def set_user(conf: omegaconf.DictConfig, user: str):
    if user is None:
        try:
            _ = conf.auth.ssh_user
        except omegaconf.MissingMandatoryValue:
            user = getpass.getuser()
            print(f'User is not specified. Continuing with the current login {user}.')
            conf.auth.ssh_user = user
    else:
        conf.auth.ssh_user = user
