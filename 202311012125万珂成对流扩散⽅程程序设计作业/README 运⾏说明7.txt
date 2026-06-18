一维对流扩散有限元程序运行说明
1. 环境依赖
运行 Python 版本：3.8 及以上
所需第三方库：numpy、matplotlib
安装命令（终端 / CMD 执行）：
pip install numpy matplotlib
加速镜像：pip install numpy matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
2. 文件清单
script10.py 主程序完整代码
README.txt 运行说明
运行后自动生成三张图片：
Pe_0.1.png Pe=0.1 数值解对比图
Pe_3.0.png Pe=3.0 数值解对比图
convergence_curve.png 网格收敛双对数曲线
3. 运行步骤
方法 1（PyCharm）
用 PyCharm 打开文件夹
右键 script10.py，选择运行
控制台输出数据，自动弹出并保存图像
方法 2（命令行）
CMD / 终端切换到代码所在文件夹
输入：python script10.py
4. 控制台输出内容
Pe=0.1 工况：单元参数、三种格式误差、全部节点数据表
Pe=3.0 工况：参数、误差、节点数据、总刚度矩阵、矩阵对称 / 正定判断
附加网格测试：nel=10/20/40/80 对应 Galerkin、SUPG 误差
5. 图像说明
Pe=0.1 图：扩散占优，三种曲线均光滑，SUPG 贴合精确解
Pe=3.0 图：对流占优，标准 Galerkin 剧烈振荡，迎风存在数值扩散，SUPG 精度最高
收敛曲线图：网格加密后两种方法误差同步下降