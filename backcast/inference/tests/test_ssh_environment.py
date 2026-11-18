"""Regression for the server's noninteractive child-session Python PATH."""
from pathlib import Path
import shlex
import shutil
import subprocess
import pytest

EXPECTED_PATH='/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'


def server_options():
    launcher=Path(__file__).resolve().parents[1]/'tools/entrypoint.sh'
    source=launcher.read_text()
    command=source.split('exec /usr/sbin/sshd ',1)[1].split('\nfi',1)[0]
    return shlex.split(command.replace('\\\n',' '))


def test_entrypoint_sets_server_child_path_without_client_environment_acceptance():
    options=server_options()
    assert 'SetEnv=PATH='+EXPECTED_PATH in options
    assert 'PasswordAuthentication=no' in options
    assert 'UsePAM=no' in options
    assert 'PubkeyAuthentication=yes' in options
    assert not any('AcceptEnv' in option or 'PermitUserEnvironment' in option for option in options)


def test_real_sshd_accepts_child_path_and_reports_effective_setting(tmp_path):
    sshd=shutil.which('sshd')
    keygen=shutil.which('ssh-keygen')
    if not sshd or not keygen:pytest.skip('OpenSSH server and keygen required')
    key=tmp_path/'ephemeral_host_key'
    subprocess.run([keygen,'-q','-t','ed25519','-N','','-f',str(key)],check=True,capture_output=True)
    config=tmp_path/'sshd_config'
    config.write_text('HostKey '+str(key)+'\nPidFile '+str(tmp_path/'sshd.pid')+'\n')
    options=server_options()
    options=[option for option in options if option not in ('-D','-e')]
    result=subprocess.run([sshd,'-T','-f',str(config),*options],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    effective=dict(line.split(' ',1) for line in result.stdout.splitlines() if ' ' in line)
    assert effective['setenv']=='PATH='+EXPECTED_PATH
    assert effective['passwordauthentication']=='no'
    assert effective['pubkeyauthentication']=='yes'
