# HP-UX Host Remote Collection

This is remote HP-UX OS monitoring on the Host object. A Linux collect node SSHes to HP-UX on each interval and runs system commands. The target does not need a collector installed. There is no version picker; version and architecture are detected automatically.

What is collected: CPU, load, memory and swap, disk capacity and inodes, disk busy and throughput, NICs, uptime, and OS version. Hardware inventory, lastlog, lsof, and connection tables are not collected.

## How to use

1. Pick a Linux collect node that can reach the HP-UX host.
2. Fill in host IP, SSH username (for example root), password or private key, and interval.
3. Save, wait one interval, and view data on Host.

## Form fields

| Field | Required | Notes |
| --- | --- | --- |
| Target Host IP | Yes | HP-UX address. |
| Username | Yes | SSH username, for example `root`. |
| SSH Authentication | Yes | Password or SSH key. |
| Password / SSH Private Key | Depends | Password auth needs a password; key auth needs a private key (passphrase optional). |
| Port | No | Default 22. |
| Collection Interval | Yes | Default 60 seconds. |
| Node | Yes | Linux node that runs collection. |

After deploy, run `plugin_init` if this plugin is not yet in the console.
