# GitHub 上传与同步指南

## 本项目配置

- 仓库：<https://github.com/MAI-Lab-West-China-Hospital/DINO-Glioma>
- 可见性：公开（2026-11-09 已设置）。提交的文件会公开展示。
- 本地路径：`/home/sun/git/DINO-Glioma`
- 远程名称：`origin`
- 远程地址：`https://github.com/MAI-Lab-West-China-Hospital/DINO-Glioma.git`
- 主分支：`main`，已跟踪 `origin/main`
- 首次上传提交：`2e0320f`，包含 17 个项目文件。

`push` 是将本地提交上传到 GitHub；`pull` 是获取 GitHub 上的提交并整合到本地。只保存文件不会自动上传，必须先 `commit`，再 `push`。

## 日常更新：先拉取，再修改，再上传

### 1. 开始修改前同步

```bash
cd /home/sun/git/DINO-Glioma
git status
git pull --ff-only origin main
```

先确认当前分支为 `main`、工作区干净。若已有未提交修改，按下文“已有本地修改时拉取”处理。`--ff-only` 会在本地与远程历史分叉时停止，不会自动生成合并提交。

### 2. 修改并检查

编辑代码或文档，保存文件，并执行与改动相关的验证。然后查看改动：

```bash
git status --short
git diff
```

### 3. 提交并上传

只添加本次准备发布的文件。例如修改了 `train.py` 和 `README.md`：

```bash
git add train.py README.md
git diff --cached
git commit -m "Update training code and documentation"
git push origin main
```

将示例文件名和提交说明替换为实际内容。新文件也需要 `git add`；删除文件可用 `git add 被删除文件的路径` 暂存删除。暂存检查时不要混入患者数据、医学影像、私密配置、访问令牌或模型权重。

### 4. 确认上传结果

```bash
git status
git log -1 --oneline
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

后两条命令输出的提交 SHA 应一致。也可打开仓库网页，查看文件和最新提交。`git status` 只反映本地已知状态，不能单独证明远程没有新提交。

## 已有本地修改时拉取

**修改已经完成，可以提交：**先按上述步骤 `git add`、`git commit`，再执行：

```bash
git pull --rebase origin main
git push origin main
```

此处 rebase 用于尚未推送的本地提交；已经共享的提交不要随意改写历史。

**修改尚未完成，暂时不想提交：**先临时保存，再同步、恢复修改：

```bash
git stash push -u -m "Work in progress before pull"
git pull --ff-only origin main
git stash pop
```

`-u` 会包含未跟踪文件，但不包含被忽略的文件。如果 `pull` 失败，先处理失败原因；`stash pop` 也可能产生冲突，需要检查后再继续工作。

## 冲突与常见问题

### 推送被拒绝：远程有新提交

在本地改动已提交且工作区干净时执行：

```bash
git pull --rebase origin main
git push origin main
```

如果 rebase 提示冲突，用 `git status` 找出冲突文件，编辑并保留正确内容，删除 `<<<<<<<`、`=======`、`>>>>>>>` 冲突标记，然后执行：

```bash
git add 实际冲突文件路径
git rebase --continue
```

按提示处理直到完成，再 `git push origin main`。若要放弃本次 rebase，执行 `git rebase --abort`。不要用 `git push --force` 代替解决冲突。

### 未登录或没有权限

```bash
gh auth status
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
```

登录时选择 HTTPS，Git 凭据验证选择 Yes，并按浏览器提示授权。账号必须具有该组织仓库的写入权限。`gh auth setup-git` 会配置 Git 使用 GitHub CLI 凭据。若其他终端显示已登录但当前环境仍未登录，用 `whoami`、`echo "$HOME"`、`gh auth status` 检查是否为同一环境；不要将 Token 写入代码、文档或聊天。

### 网络错误：EOF、TLS handshake failed

本次上传遇到过间歇性连接中断，重试后成功。这类错误不一定是账号权限问题。检查网络后重试 `git push origin main`，不要关闭 TLS 证书验证。

若 Git 未正确使用已登录的 GitHub CLI 凭据，可仅对本次推送指定凭据助手：

```bash
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin main
```

## 在另一台电脑上获取项目

```bash
git clone https://github.com/MAI-Lab-West-China-Hospital/DINO-Glioma.git
cd DINO-Glioma
```

公开仓库的克隆和拉取不需要写入权限；上传仍需登录有写入权限的账号。新电脑首次提交前，设置自己的提交身份，例如：

```bash
git config user.name "你的 GitHub 用户名"
git config user.email "你的 GitHub 已验证邮箱或 noreply 邮箱"
```

## 首次上传流程记录

以下步骤已在本项目完成，日常更新无需重复初始化或创建仓库：

1. 检查项目文件、Git 状态与 GitHub 登录状态。
2. 使用 `sunhuaiqiang` 账号验证组织权限，在组织下创建私有 `DINO-Glioma` 仓库。
3. 初始化本地 Git 仓库和 `main` 分支，设置提交身份，添加远程 `origin`。
4. 显式暂存项目源码、README 和流程图，创建首次提交。
5. 执行 `git push -u origin main`，成功上传并建立分支跟踪。
6. 按要求将仓库改为公开，并通过 GitHub API 确认可见性为 `public`。

首次初始化的命令结构如下，仅供新项目参考：

```bash
git init -b main
git remote add origin https://github.com/MAI-Lab-West-China-Hospital/DINO-Glioma.git
git add README.md workflow.png train.py test.py data evaluation helpers losses model
git commit -m "Initial commit: DINO-Glioma source code"
git push -u origin main
```

运行前须已创建相应远程仓库并设置提交身份。本项目已经有 `origin`，不要重复执行 `git remote add origin`。
