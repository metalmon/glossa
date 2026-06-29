# Ansible install — glossa from GitHub Release

Installs `kb` from a release tarball, creates a corpus directory, runs an initial index, and registers a **systemd** service (`glossa-mcp`) on loopback HTTP.

## Requirements

- Ansible 2.14+
- Target: Linux x86_64 (`x86_64-unknown-linux-gnu` release asset)
- Root/sudo on the target host

## Quick start

```bash
cd deploy/ansible
ansible-playbook -i inventory playbook.yml \
  -e glossa_version=1.0.0 \
  -e glossa_corpus_path=/srv/glossa/corpus
```

## Variables

Set in `group_vars/all.yml` or pass with `-e`:

| Variable | Default | Description |
|----------|---------|-------------|
| `glossa_version` | `1.0.0` | Release tag without `v` |
| `glossa_corpus_path` | `/srv/glossa/corpus` | Document folder (must be readable by `glossa_user`) |
| `glossa_profile` | `reader` | MCP profile: `reader`, `editor`, or `full` |
| `glossa_bind` | `127.0.0.1:8080` | HTTP bind address |
| `glossa_allowed_host` | `localhost` | `--allowed-host` for MCP |
| `glossa_install_dir` | `/opt/glossa` | Install root |
| `glossa_user` | `glossa` | Service account |

## After install

- MCP endpoint: `http://127.0.0.1:8080/mcp` (adjust if `glossa_bind` changed)
- Health: `curl http://127.0.0.1:8080/health`
- Put documents in `glossa_corpus_path`, then `sudo -u glossa /opt/glossa/bin/kb index /srv/glossa/corpus` if you add files before auto-index runs

Connect agents: [docs/connect-to-agents.md](../../docs/connect-to-agents.md)

## Uninstall

```bash
sudo systemctl disable --now glossa-mcp
sudo rm /etc/systemd/system/glossa-mcp.service
sudo systemctl daemon-reload
sudo rm -rf /opt/glossa
```

Corpus files under `glossa_corpus_path` are not removed.
