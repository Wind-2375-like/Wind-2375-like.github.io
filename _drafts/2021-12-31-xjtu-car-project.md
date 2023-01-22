---
title: '树莓派小车环境搭建与连接 (某 C9)'
date: 2021-12-31
permalink: /posts/2021/12/xjtu-car-project/
excerpt_separator: <!--more-->
toc: true
tags:
  - diary
---

<div style="display:none">
大四了，除了保研的，我们这帮考研的出国的一堆b事，还得干这个干那个，真是淦啊。
</div>

花点时间写个文档，把其他人写过的整合一下，再把自己踩过的坑补充一下。

<!--more-->

要求：车，装好了（接线也很麻烦）；给的那堆资料都在；装过给的那个用着win10桌面的树莓派系统 (笑死) 测试过电机，摄像头等传感器都正常

## 环境搭建

### 安装PI4系统

首先自己搞一个pi4系统然后安装啥的，就照着这个来。

[如何启用WIFI进行树莓派的首次连接 | Sonic Guo Blog](http://sonicguo.com/2019/Initialize-RaspBerry-3-with-wifi/)

#### 前期准备

刚刚打开树莓派的时候，我是懵逼的。连电源线该插在哪里，SD卡的插入的方向等问题都比较疑惑。赶紧找来主板的结构体看了一眼，压压惊。
![mainboard](http://sonicguo.com/2019/Initialize-RaspBerry-3-with-wifi/mainboard.png)

主要硬件：

1. 树莓派3B+主板（其他型号过程类似）。
2. 一块SD卡，用来安装Raspbian。
3. 供电电源及Micro USB线（注意电源的质量和功率也很重要，最好用那种带开关的线）。说明书上的安全指南指出，==本产品仅可连接到额定功率为5V、最小电流为2.5Amp的直流电外接电源。==所以在接入电源前，要先检查一下电源的输出功率。==（电池供电10来分钟就gg了，所以我用5V 3A充电电流的插头配mini usb接口的线连接树莓派板子）==

#### 制作Raspbian的SD卡

SD卡将包含Raspberry Pi的操作系统（操作系统是一种使Raspberry Pi工作的软件，就像PC里的Windows和Mac里的OSX）。这个操作系统与大部分电脑的系统有很大的不同。由于我使用的是Windows操作系统，所以这里的步骤是记录Windows下的操作步骤。

1. **下载 Raspberry Pi 操作系统** ：[https://www.raspberrypi.org/downloads/](https://www.raspberrypi.org/downloads/raspbian/)
   这里一共有三个版本，为了图省事，我安装的是 ==Raspberry Pi OS with desktop and recommended software==
   下载完成后，解压镜像，以备后用。这个 .img文件只能用专用的软件写入SD卡中。接下来需要使用另外一个工具烧录镜像到SD卡中。
2. **Win32DiskImager下载** : https://sourceforge.net/projects/win32diskimager/
   下载并且安装好以后就可以开始烧录SD卡了。
3. **将Raspbian 镜像写入 SD卡** : ==在这一步之前，如果卡已经装着给的那个Hjduino系统，就用格式化工具sdformatter格式化，再写入！==打开Win32DiskImager。在Image File上选中解压出来的Raspbian Image. Device指向SD所在的盘符。点击Write就开始烧录了。整个过程可能需要几分钟就可以完成。
   ![win32diskimager](http://sonicguo.com/2019/Initialize-RaspBerry-3-with-wifi/win32diskimager.png)

#### 配置WIFI

由于我的环境是没有额外的显示器，鼠标键盘，甚至没有额外的网线。所以需要第一次启动之后就使用WIFI的方式连接到树莓派上。因此需要现在SD卡上配置好WIFI。首先在SD卡的根目录下添加一个名为 wpa_supplicant.conf的文件，然后在该文件内添加以下的内容 ：

```
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
network={
  ssid="WIFI名"
  psk="WIFI密码"
}
```

==这个wifi名连接宿舍的2.4Gwifi，密码连宿舍的wifi密码，你的电脑也得连，目的是让小车和你的电脑共用一个网==

插入SD卡启动树莓派就能直接连接到你的WIFI网络了（切记树莓派现时只支持802.11.n的WIFI标准所以只能连接2.4G网络，所以你需要确保你所连接的是2.4G的通道而不是5G的。

#### 启用SSH

Raspbian 默认情况下是将SSH服务关闭的。开启SSH的方法很简单在树莓的官网上也有介绍，只要在新建一个名为==SSH==文件（我用的是大写，我也不知道小写行不行）到SD的根目录就能完成。这里要注意的是，新建的是文件，并且确保去掉后缀名。而不是新建SSH的文件夹。
![ssh](http://sonicguo.com/2019/Initialize-RaspBerry-3-with-wifi/ssh.png)

#### 查找IP

配置好SD卡之后，将SD插入树莓派，接入电源就能正常启动了。要想连接到树莓派首先要找它的IP地址。我使用的ASUS路由器，可以登陆到路由器上直接看到对应的IP。

#### 使用PuTTY 连接到树莓派上

在这里可以[Download PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latespit.html)。
安装好之后打开PuTTY，输入上一步找到的IP地址就可以连进去。默认的用户名是`pi`，默认密码是`raspberry`
![PuTTY](http://sonicguo.com/2019/Initialize-RaspBerry-3-with-wifi/PuTTY.png)
![login](http://sonicguo.com/2019/Initialize-RaspBerry-3-with-wifi/login.png)

#### 使用VNC Viewer连接到树莓派

使用PuTTY连接到树莓派上，就可以开始安装VNCServer, 从而使用界面了。可以直接使用命令来安装你的VNCServer。

```shell
sudo apt-get install tightvncserver
```

成功安装以后，就可以在终端中输入”vncserver”来启动你的远程界面服务器。登陆时会提输入vncserver的密码，输入确认以后会看见类似下面的内容。这样vncserver就成功启动了。

==看好了，端口号是1==

==以后连接vnc之前，必须通过putty输入vncserver开vnc！！！==

![vncserver](http://sonicguo.com/2019/Initialize-RaspBerry-3-with-wifi/vncserver.png)

可以从这里[下载VNC Viewer](https://www.realvnc.com/en/connect/download/viewer/)
下载好以后，在VNC Viewer中输入树莓派的IP，以及刚才设置好的密码，登陆到树莓派上。

![vncviwer](http://sonicguo.com/2019/Initialize-RaspBerry-3-with-wifi/vncviwer.png)

登录之后，按照引导教程来，到更新软件那一步，==一定skip！！！==，等不起连不上！

### 安装opencv(C++)

就照着这个来[树莓派4 安装OPENCV3全过程（各种踩坑和报错）_肿么阔以次兔兔的博客-CSDN博客_树莓派安装opencv](https://blog.csdn.net/weixin_43287964/article/details/101696036?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522160914304416780273327251%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=160914304416780273327251&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-101696036.first_rank_v2_pc_rank_v29&utm_term=树莓派安装OpenCV)

#### 更换源

第一步先更换源，更新下载更快；
①在终端输入以下指令

```
sudo nano /etc/apt/sources.list
```

用#注释掉原文件内容，用以下内容取代：

```
deb http://mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian/ buster main contrib non-free rpi
deb-src http://mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian/ buster main contrib non-free rpi
```


这里我用的是是清华源，在后面若下载出错换中科大源试试（Ps：中科大的源地址http://mirrors.ustc.edu.cn/raspbian/raspbian/
对应替换就行)

注意：网上很多是 stretch ，这是以前的版本 ， 现在已经改成 buster了，之前博主就是因为这个无限黑屏。
所以如果你的树莓派版本较新的话就用我这个代码就行，老版本的自行备份测试。
如图：

![图1](https://img-blog.csdnimg.cn/20190929152655322.png)然后ctrl+o保存，点回车确认保存，然后ctrl+x退出

②在终端输入以下指令**

```
sudo nano /etc/apt/sources.list.d/raspi.list
```


用#注释掉原文件内容，用以下内容取代：

```
deb http://mirrors.tuna.tsinghua.edu.cn/raspberrypi/ buster main ui
deb-src http://mirrors.tuna.tsinghua.edu.cn/raspberrypi/ buster main ui
```

如图：

![图2](https://img-blog.csdnimg.cn/20190929153617230.png)

然后ctrl+o保存，点回车确认保存，然后ctrl+x退出

③使用命令更新软件源列表，同时检查编辑是否正确。再更新软件

```
sudo apt-get update
sudo apt-get upgrade
```

#### 存储空间的一些说明和操作

必须使用16G以上的卡，最好是class10以上，实测8G class6卡安装到35%就爆满了，推算安装完成要4.6G左右。
然后扩大文件系统。因为，用SD卡安装完系统后一大部分空间实际是未被分配的
使用命令

```
sudo raspi-config
```


然后选择==advanced option==

![图3](https://img-blog.csdnimg.cn/20190929154701492.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzI4Nzk2NA==,size_16,color_FFFFFF,t_70)

然后选择第一个回车，会让你重启树莓派，选择立即重启

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929154909423.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzI4Nzk2NA==,size_16,color_FFFFFF,t_70)

#### 增加交换空间

增加交换空间以避免因内存问题导致的编译挂起
输入命令

```
sudo nano /etc/dphys-swapfile
```

将 CONF_SWAPSIZE 值从默认值更改 100 为 1024 ：
如图

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929160016354.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzI4Nzk2NA==,size_16,color_FFFFFF,t_70)

然后ctrl+o保存，点回车确认保存，然后ctrl+x退出，并运行以下命令以使更改生效：

```
sudo /etc/init.d/dphys-swapfile restart
```

#### 下载工具及包

①安装OpenCV的相关工具

```
sudo apt install build-essential cmake git pkg-config libgtk-3-dev libcanberra-gtk*
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev
sudo apt install libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev opencl-headers
sudo apt install python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev
```

PS：如果出现（无法定位软件包)啥的，换个源更新一下试试，建议试试中科大的源，更新快。因为软件一直在更新，不知道此刻以后会出现什么奇奇怪怪的版本，具体怎么怎么换上面也写了，只是网站不一样，百度一大堆

②创建一个新目录并从 Github 克隆 OpenCV 和 OpenCV contrib 存储库：

```
mkdir ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```


因为Github服务器不在国内，并且不支持断点续传，偶尔抽风很正常，克隆失败多半就是克隆过程中抽风了，多试几次，实在不行就自己上墙，下载好了再拷贝到树莓派上，别再私信我这种问题哥哥们，百度！百度！百度！百度不了就谷歌一下

在撰写本文时， GitHub 存储库中的默认版本是 4.5.4 版。如果你想安装 OpenCV 的旧版本，导航既 opencv 和 opencv_contrib 目录，并运行 git checkout

克隆存储库后，创建一个临时构建目录，然后切换到该目录：

```
mkdir -p ~/opencv_build/opencv/build
cd ~/opencv_build/opencv/build
```

#### 设置编译编译参数

==注意，一定要一行一行复制输入，否则会有回车，在键入中会直接运行，这样会报错。==

```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D ENABLE_NEON=ON \
    -D ENABLE_VFPV3=ON \
    -D BUILD_TESTS=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=OFF ..
```

“\” 代表将代码延续到下一行
输出结果如下所示：

```
...
-- Configuring done
-- Generating done
-- Build files have been written to: /home/pi/opencv_build/opencv/build
```

#### 开始编译

运行

```
make -j4
```

这个过程大概需要两三个小时，耐心等，不要用树莓派做其他事

==芯片会很烫，开风扇吧==

编译结束了会出现下面这个结果

```
...
[100%] Linking CXX shared module ../../lib/python3/cv2.cpython-35m-arm-linux-gnueabihf.so
[100%] Built target opencv_python3
```

**如果编译在某些时候失败，由于资源不可用，请 make 再次运行该命令，该过程将从停止的位置继续。**

**最开始用`make -j4`**，比较快。后面会卡99%，这时候ctrl+C，用`make`完成后面的。

成功后安装已编译的 OpenCV 文件：

```
sudo make install
```

时间很快，结果如下

```
...
-- Installing: /usr/local/bin/opencv_version
-- Set runtime path of "/usr/local/bin/opencv_version" to "/usr/local/lib"
```


最后检查opencv安装成功与否

C++库：

```
pkg-config --modversion opencv4
```


结果会输出版本号。

#### 收尾

如果 SD 卡上没有足够的可用空间，请删除源文件：

```
rm -rf ~/opencv_build
```

大量交换使用可能会损坏您的 SD 卡。将交换空间更改回原始大小：

```
sudo nano /etc/dphys-swapfile
```


将 CONF_SWAPSIZE 值改回 100

保存文件并激活更改：

```
sudo /etc/init.d/dphys-swapfile restart
```


开始使用opencv吧！

### 安装Qt并配置摄像头

就照着这个来 [课程项目设计——HJduino树莓派小车0基础运行开发_h_lonely的博客-CSDN博客](https://blog.csdn.net/h_lonely/article/details/109174581)

#### 安装qt

```
sudo apt-get install qt5-default
sudo apt-get install qtcreator
```

安装完毕后在树莓派的主界面可以找到他，也可以拖动到桌面快捷方式。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210105223301144.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hfbG9uZWx5,size_16,color_FFFFFF,t_70)

新建QT工程，按下图选择Non-QT项目

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210105223524652.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hfbG9uZWx5,size_16,color_FFFFFF,t_70)


测试运行，成功build。不知为何debug失败。

#### 摄像头

##### 添加驱动程序文件

```
sudo nano /etc/modules
```

我们是没有装vim的，所以nano直接打开该文件，在最后一行加入

```
bcm2835-v4l2
```

ctrl+o写入回车并ctrl+x保存

##### 修改Raspberry的启动配置使能项：

```
sudo raspi-config
```

![img](https://images2017.cnblogs.com/blog/772331/201709/772331-20170924160535118-1858246368.png)

得到如下的配置界面：

![img](https://images2017.cnblogs.com/blog/772331/201709/772331-20170924160732743-1394685215.png)

选择==Interface Option==，选中Select然后Enter进入，如下图所示：

![img](https://images2017.cnblogs.com/blog/772331/201709/772331-20170924160831040-2092075408.png)

接下来机会问你是否同意使能Pi camera，选择是然后会让你重启，，重启就好了：

![img](https://images2017.cnblogs.com/blog/772331/201709/772331-20170924160938243-536503361.png) 选择 “是”

执行下面的指令

```
raspistill -o image.jpg
```

在运行目录打开图片，看看可不可以。

### 抄代码前的配置环境

打开项目工程文件.pro，改

==注意LIBS里面的目录自己查一下看看版本号==

==注意要有LIBS += -liwiringPi==

==注意查includepath里面的具体地址==

```
QT      += core

QT      -= gui

TARGET   = picproc

CONFIG  += console c++11

CONFIG  -= app_bundle

TEMPLATE = app

SOURCES += main.cpp

LIBS    +=  /usr/local/lib/libopencv_highgui.so.4.5\
            /usr/local/lib/libopencv_imgproc.so.4.5\
            /usr/local/lib/libopencv_imgcodecs.so.4.5\
            /usr/local/lib/libopencv_core.so.4.5\
            /usr/local/lib/libopencv_shape.so.4.5\
            /usr/local/lib/libopencv_videoio.so.4.5\
            /usr/local/lib/libopencv_calib3d.so.4.5\
            /usr/local/lib/libopencv_features2d.so.4.5\
            /usr/local/lib/libopencv_ml.so.4.5\
            /usr/local/lib/libopencv_objdetect.so.4.5\
            /usr/local/lib/libopencv_photo.so.4.5\
            /usr/local/lib/libopencv_stitching.so.4.5\
            /usr/local/lib/libopencv_video.so.4.5\

/usr/local/lib/libopencv_imgcodecs.so.4.5

LIBS += -lwiringPi

INCLUDEPATH +=  /usr/local/include/opencv4

```

然后就抄代码就完事了！

### 离开宿舍连树莓派

共享手机热点

写入wpa文件在network追加一个手机热点的，重启树莓派。

非华为手机，下载Android terminal，在里面输入ip neigh，找到树莓派的，在putty里面连接。

## 小车连接

### 前期准备

- micro usb线, 梯形的那个
- 5V, 电流2.5A以上的充电头
- [vnc viewer安装]([Download VNC Viewer | VNC® Connect (realvnc.com)](https://www.realvnc.com/en/connect/download/viewer/))(选择Windows,standalone exe x64)
- putty中文版, 有
- filezilla, 有
- 能开热点的安卓手机, 检查能否查看连接手机设备的ip地址, 若不能, 需要下载[Termux]([高级终端Termux(com.termux) - 0.73 - 应用 - 酷安 (coolapk.com)](https://www.coolapk.com/apk/com.termux))
- 两节26650 3.7V可充电锂电池, 最好容量大一点, 现在没有
- **可选**: 黑胶带(东南门小吃街走到第一个路口右转, 靠右手边有一家五金商店), 和sd卡读卡器(康三), 都是几块钱的东西.

### 共同联网

首先需要让小车和电脑连接同一个网络.

在boot根目录下添加一个名为 wpa_supplicant.conf的文件，然后在该文件内添加以下的内容 ：

```shell
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
network={
  ssid="WIFI名"
  psk="WIFI密码"
}
network={
  ssid="WIFI名"
  psk="WIFI密码"
}
network={
  ssid="WIFI名"
  psk="WIFI密码"
}
# 想写多少写多少
```

这里wifi名, wifi密码对应手机热点 or 宿舍wifi, 如果宿舍的话, ==不要==用5G频段的网络.

sd卡插到树莓派一上电, 这个文件就会覆盖掉系统内(插到windows系统电脑看不见) `/etc/wpa_supplicant/wpa_supplicant.conf`的文件, 要注意. 我会直接用sd卡读卡器写入所有人的手机热点网络, 后期也可以用宿舍网, 但添加的时候最好不要覆盖原文件.

### 查找IP地址

小车和电脑连接一个手机热点之后, 需要查连接手机设备的ip地址. 发现好像就华为自带这个功能, 其他的需要用Termux, 手机内的命令行程序, 去查找. 下载好之后, 命令行输入

 `````shell
 ip neigh
 `````

找到物理地址以B开头的, 或者名字有rasp字样的, 记录ip地址.

### 连接putty

打开putty, 输入刚才的ip地址

![image-20211024161741925](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202110241617278.png)

然后打开命令行

```
login as: pi
pi@192.168.1.2's password: 12345678
```

然后输入

```bash
vncserver
```

得到:

![image-20211024162929153](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202110241629323.png)

查看New 'X' 那一行, 显示 ":1", 记住.

### 连接VNCViewer

输入ip地址, 加上端口. 注意冒号要用英文的.

![image-20211024163118369](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202110241631057.png)

### 打开项目

打开qt creator

![image-20211024163324272](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202110241633246.png)

然后【文件】-【打开文件或项目】-找 `/home/pi/car_project/car_project.pro

找到main.cpp, 运行, 如果想让小车跑, 得需要上电池

![image-20211024163609026](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202112211911833.png)

代码是抄的 [课程项目设计——HJduino树莓派小车0基础运行开发_h_lonely的博客-CSDN博客](https://blog.csdn.net/h_lonely/article/details/109174581)

```c++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <softPwm.h>
#include <wiringPi.h>

using namespace std;
using namespace cv;

#define RUNSPEED 30
#define TURNSPEED 35
void softpwminit()   //初始化pwm波，涉及到wiringpi库
{
    // initial the pwm for 4 wheels
    // 500ms as main period
    // the following function's parameters are set as n% of speed,100% is faster level;

    wiringPiSetup();
    pinMode (1, OUTPUT);	//IN1   //初始化1,4,5,6号引脚为输出模式
    pinMode (4, OUTPUT);	//IN2
    pinMode (5, OUTPUT);	//IN3
    pinMode (6, OUTPUT);	//IN4
    softPwmCreate(1,1,500);  // 1号引脚pwm波周期为500ms
    softPwmCreate(4,1,500);
    softPwmCreate(5,1,500);
    softPwmCreate(6,1,500);

}

void run(int n)     // 前进函数,100 is maxium.
{
    softPwmWrite(4,0); //左轮前进
    softPwmWrite(1,5*n);
    softPwmWrite(6,0); //右轮前进
    softPwmWrite(5,5*n);
    printf("run");
    delay(100);
}

void brake()         //刹车，停车
{
    softPwmWrite(1,0); //左轮stop
    softPwmWrite(4,0);
    softPwmWrite(5,0); //stop
    softPwmWrite(6,0);
    printf("brake");
    //delay(200);
}

void left(int n)         //左转()
{
    softPwmWrite(4,5*n); //左轮后退
    softPwmWrite(1,0);
    softPwmWrite(6,0); //右轮前进
    softPwmWrite(5,5*n);
    printf("left");
    delay(100);
}

void right(int n)        //右转()
{
    softPwmWrite(4,0); //左轮前进
    softPwmWrite(1,5*n);
    softPwmWrite(6,5*n); //右轮
    softPwmWrite(5,0);
    printf("right");
    delay(100);
}

void back(int n)          //后退
{
    softPwmWrite(4,5*n); //左轮back
    softPwmWrite(1,0);
    softPwmWrite(6,5*n); //右轮back
    softPwmWrite(5,0);
    printf("back");
    delay(100);     //执行时间，可以调整
}
int main()
{
    VideoCapture vcap(0);
    vcap.set(CAP_PROP_FRAME_WIDTH, 320);
    vcap.set(CAP_PROP_FRAME_HEIGHT, 240);
    // Scalar sca;
    Mat f1, f2;
    int run_flag=1,black_row=120,right_detected=0,left_detected=0,right_danger=0,left_danger=0;
    int danger_row;
    softpwminit();
    int center_black=0;
    brake();

    while(1)
    {
        vcap >> f1;
        cvtColor(f1, f1, COLOR_BGR2GRAY);
        threshold(f1,f2,60,255,THRESH_BINARY);
        imshow("chengzige",f2);

        for(int i = 210,j = 0;i>=118;i--)
        {
            if(f2.ptr<uchar>(i)[160] == 0) j++;
            if (j>=4)
            {
                black_row = i;
                center_black=1;
                j=0;
                break;
            }
        }
        
        for(int i = 20,j = 0;i <300;i++)
        {
            if(f2.ptr<uchar>(239)[i] == 0) j++;
            if (j>=3)
            {
                danger_row = i;
                if (danger_row > 160)
                {
                    right_danger = 1;
                }
                else if(danger_row < 160)
                {
                    left_danger = 1;
                }
                break;
            }
        }


        if(center_black)
        {
            for(int i =160,j=0;i<=320;i++)
            {
                if(f2.ptr<uchar>(black_row+25)[i]==0)
                {
                   j++;
                }
                if(j>=3)
                {
                    right_detected = 1;
                    break;
                }
            }

            for(int i =160,j=0;i>0;i--)
            {
                if(f2.ptr<uchar>(black_row+25)[i]==0)
                {
                   j++;
                }
                if(j>=3)
                {
                    left_detected = 1;
                    break;
                }
            }

            if (left_detected&&right_detected)
            {
                back(RUNSPEED);
                left_detected = 0;
                right_detected = 0;
                center_black = 0;
            }
            else if (right_detected)
            {
                left(TURNSPEED);
                left_detected = 0;
                right_detected = 0;
                center_black = 0;
            }
            else
            {
                right(TURNSPEED);
                left_detected = 0;
                right_detected = 0;
                center_black = 0;
            }
        }
        else
        {
            if (left_danger)
            {
                right(TURNSPEED);
                cout<<"danger_right"<<endl;
                left_danger = 0;
            }
            else if (right_danger)
            {
                left(TURNSPEED);
                cout<<"danger_left"<<endl;
                right_danger = 0;
            }
            else
            {
                run(RUNSPEED);
                cout<<"else_run"<<endl;
            }
        }

        //show f2
        if(char(waitKey(1)) == 'q')
        {
           brake();
           break;
        }
    }

    vcap.release();
    return 0;
}
```

### Filezilla使用

vncviewer不能同步文件和剪贴板, 只能借助这东西文件传输:

![image-20211024163800210](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202110241638949.png)

### 项目代码

```c++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <softPwm.h>
#include <wiringPi.h>

using namespace std;
using namespace cv;

#define RUNSPEED 30
#define TURNSPEED 35
#define MG 40           // 左右边界
#define SMG 10          // 小边界

void softpwminit()      // 初始化 4轮 PWM, 涉及到wiringpi库
{
    wiringPiSetup();
    
    // 初始化1, 4, 5, 6号引脚为输出模式
    pinMode (1, OUTPUT);		//IN1   
    pinMode (4, OUTPUT);		//IN2
    pinMode (5, OUTPUT);		//IN3
    pinMode (6, OUTPUT);		//IN4
    
    // 1, 4, 5, 6号引脚pwm波周期为500ms
    softPwmCreate(1, 1, 500);  
    softPwmCreate(4, 1, 500);
    softPwmCreate(5, 1, 500);
    softPwmCreate(6, 1, 500);
}

void run(int n)         // 前进函数, 以n%的速度行驶.
{
    // 左轮前进
    softPwmWrite(4, 0); 
    softPwmWrite(1, 5*n);
    
    // 右轮前进
    softPwmWrite(6, 0); 
    softPwmWrite(5, 5*n);
    
    printf("run");
    delay(100);
}

void brake()            // 刹车，停车
{
    // 左轮stop
    softPwmWrite(1, 0); 
    softPwmWrite(4, 0);
    
    // 右轮stop
    softPwmWrite(5, 0);
    softPwmWrite(6, 0);
    
    printf("brake");
}

void left(int n)        // 左转
{
    // 左轮后退
    softPwmWrite(4, 5*n); 
    softPwmWrite(1, 0);
    
    // 右轮前进
    softPwmWrite(6, 0); 
    softPwmWrite(5, 5*n);
    
    printf("left");
    delay(100);
}

void right(int n)       //右转
{
    // 左轮前进
    softPwmWrite(4, 0); 
    softPwmWrite(1, 5*n);

    // 右轮后退
    softPwmWrite(6, 5*n); 
    softPwmWrite(5, 0);

    printf("right");
    delay(100);
}

void back(int n)        //后退
{
    // 左轮后退
    softPwmWrite(4, 5*n);
    softPwmWrite(1, 0);

    // 右轮后退
    softPwmWrite(6, 5*n);
    softPwmWrite(5, 0);

    printf("back");
    delay(100);
}

int main()
{
    // 设置画布像素大小
    VideoCapture vcap(0);
    vcap.set(CAP_PROP_FRAME_WIDTH, 320);
    vcap.set(CAP_PROP_FRAME_HEIGHT, 240);

    Mat f1, f2;         // f1存储灰度图, f2存储二值化黑白图
    int run_flag = 1;
    int left_black_row = 120, right_black_row = 120;
    int left_black = 0, right_black = 0;
    int left_detected = 0, right_detected = 0;
    int left_danger = 0, right_danger = 0;
    int danger_row;

    softpwminit();
    brake();

    while (1)
    {
        // 得到二值化黑白图
        vcap >> f1;
        cvtColor(f1, f1, COLOR_BGR2GRAY);
        threshold(f1, f2, 60, 255, THRESH_BINARY);
        imshow("vision",f2);

        // 检测左侧是否有交点
        for (int i = 210, jl = 0; i >= 118; i--)
        {
            if (f2.ptr<uchar>(i)[160-MG] == 0) jl++;
            if (jl >= 4)
            {
                // 记录交点对应的行
                left_black_row = i;

                left_black = 1, jl = 0;
                break;
            }
        }

        // 检测右侧是否有交点
        for (int i = 210, jl = 0; i >= 118; i--)
        {
            if (f2.ptr<uchar>(i)[160+MG] == 0) jr++;
            if (jr >= 4)
            {
                // 记录交点对应的行
                right_black_row = i;  

                right_black = 1, jr = 0;
                break;
            }
        }
        
        // 都没有交点, 可能直行, 判断是否偏离跑道
        if (!left_black && !right_black)
        {
            for(int i = 20, j = 0; i < 300; i++)
            {
                if(f2.ptr<uchar>(239)[i] == 0) j++;
                if (j >= 3)
                {
                    danger_row = i;
                    if (danger_row > 160)
                    {
                        right_danger = 1;
                    }
                    else if(danger_row < 160)
                    {
                        left_danger = 1;
                    }
                    break;
                }
            }

            if (left_danger)
            {
                left(TURNSPEED);
                cout << "danger: turn left" << endl;
                left_danger = 0;
            }
            else if (right_danger)
            {
                right(TURNSPEED);
                cout << "danger: turn right" << endl;
                right_danger = 0;
            }
            else
            {
                run(RUNSPEED);
            }
        }

        // 都有交点, 直接直行
        else if (left_black && right_black)
        {
            run(RUNSPEED);
            left_black = 0;
            right_black = 0;
        }

        // 左侧有交点
        else if (left_black)
        {
            for(int i = 160-MG+SMG, j = 0; i <= 320; i++)
            {
                if(f2.ptr<uchar>(left_black_row)[i] == 0)
                {
                   j++;
                }

                // 检测是否出现第二交点
                if(j >= 3)
                {
                    right_detected = 1;
                    break;
                }
            }

            if (right_detected) run(RUNSPEED);
            else left(TURNSPEED);

            left_detected = 0;
            right_detected = 0;
            left_black = 0;
            right_black = 0;
        }

        // 右侧有交点
        else if (right_black)
        {
            for(int i = 160+MG-SMG, j = 0; i > 0; i--)
            {
                if(f2.ptr<uchar>(right_black_row)[i]==0)
                {
                   j++;
                }

                // 检测是否出现第二交点
                if(j >= 3)
                {
                    left_detected = 1;
                    break;
                }
            }

            if (left_detected) run(RUNSPEED);
            else right(TURNSPEED);

            left_detected = 0;
            right_detected = 0;
            left_black = 0;
            right_black = 0;
        }

        // 刹车
        if(char(waitKey(1)) == 'q')
        {
            brake();
            break;
        }
    }

    vcap.release();
    return 0;
}
```