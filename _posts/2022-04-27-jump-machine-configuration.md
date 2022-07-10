---
title: 'Vscode configuration for jump machine'
date: 2022-04-27
permalink: /posts/2022/04/jump-machine-configuration/
excerpt_separator: <!--more-->
toc: true
tags:
  - diary
---

How can I connect to a server that only allows specific ip (e.g. lab intranet) logins without connecting to this specific ip?

<!--more-->

If you connect to another server that can connect to your real target server, you can set this server up as a jump machine.

## Generate ssh public/private rsa key pair (Windows)

(Git bash)

```shell
cd ~/.ssh
ssh-keygen -t rsa
```

## Upload public key to target machine and jump machine (Windows)

(Git bash)

```shell
ssh-copy-id -i ~/.ssh/id_rsa.pub [jump_machine]@[jump_machine_ip]
ssh-copy-id -i ~/.ssh/id_rsa.pub [target_machine]@[target_machine_ip]
```

## Edit ~/.ssh/config

```python
Host JumpMachine
  HostName [jump_machine_ip]
  User [jump_machine]
  IdentityFile "C:\Users\[name]\.ssh\id_rsa"

Host TargetMachine
  HostName [target_machine_ip]
  User [target_machine]
  IdentityFile "C:\Users\[name]\.ssh\id_rsa"
  ProxyCommand ssh -W %h:%p JumpMachine
```

## On Vscode ssh

<img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/eb08724735e73e265359323e03c154bf6bf36ab01c3c163d22769a6e294bb875.png" />  

<script src="https://utteranc.es/client.js"
    repo="Wind-2375-like/Wind-2375-like.github.io"
    issue-term="pathname"
    theme="github-light"
    crossorigin="anonymous"
    async>
</script>
