# EEVEE报错
既然 EGL 找不到合适的显示 Surface，就用 Xvfb 凭空给它造一个：

先安装：sudo apt-get install xvfb

运行你的 Python 脚本：xvfb-run -a python your_script.py
(很多 Docker 用户反馈，加上这句后 EGL_BAD_MATCH 报错就消失了)