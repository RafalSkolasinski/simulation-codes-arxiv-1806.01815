import os
from datetime import datetime
from subprocess import check_output


def check_repo_folder():
    fname = os.path.realpath(__file__)
    folder = os.path.dirname(fname)

    cmds = ['git', 'rev-parse', '--show-toplevel']
    repo_folder = check_output(cmds, cwd=folder, universal_newlines=True)
    return repo_folder.split("\n")[0]


def check_git_commit():
    folder = check_repo_folder()

    def check_git_version():
        cmds = ['git', 'rev-parse', 'HEAD']
        ver = check_output(cmds, cwd=folder, universal_newlines=True)
        return ver.split("\n")[0]

    def check_branch():
        cmds = ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        branch = check_output(cmds, cwd=folder, universal_newlines=True)
        return branch.split("\n")[0]

    def check_code_status():
        cmds = ['git', 'diff', 'codes']
        res1 = check_output(cmds, cwd=folder, universal_newlines=True)

        cmds = ['git', 'diff', '--staged', 'codes']
        res2 = check_output(cmds, cwd=folder, universal_newlines=True)

        return 'modified' if (res1 or res2) else 'clean'

    res = {
        'commit': check_git_version(),
        'branch': check_branch(),
        'code_status': check_code_status()
    }
    return res


def check_machine():
    machine = check_output(['uname', '-n'], universal_newlines=True)
    return machine.split("\n")[0]


def check_date():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def check_status():
    out = check_git_commit()
    out.update({'machine': check_machine(), 'date': check_date()})
    return out
