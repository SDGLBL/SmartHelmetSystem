# SmartHelmetSystem

​        **此项目只用于进行安全帽目标探测，不包括配套的推流和前端项目，如需要整个项目请自行下载并按照技术文档中的部署说明部署**

- [整套项目下载](https://1drv.ms/u/s!Ag3iZuii89kugstA-5_21EYW7IQt8A?e=tmC0Xo)
- [技术文档](https://1drv.ms/u/s!Ag3iZuii89kugsslK_IF6mIKcEwmog?e=Sr1OUl)

### 本检测项目部署

**要求环境**

- Ubuntu18.04
- CUDA == 10.1
- CUDNN
- gcc and g++ >= 7.5

**下载权重并放置到根目录**

- [权重](https://1drv.ms/u/s!Ag3iZuii89kugt92sO1vpO42ZrmwFw?e=llmhQt)

**安装运行环境**

```bash
# 安装 virtualenv
sudo pip3 install -U virtualenv
# 创建 virtual env
virtualenv --system-site-packages -p python3 ./venv
# 激活 virtual env
source venv/bin/activate
# 开始安装与编译依赖
python detection/setup.py develop
```

**测试视频**

```bash
python tools/process_video.py detection/myconfigs/baseline/faster_rcnn_r50_fpn_1x.py fastr50_963.pth --input_path [video path] --output_path [save_path]
```

**引用**

- [mmdetection](https://github.com/open-mmlab/mmdetection)