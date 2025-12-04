# ComfyUI 图片处理节点

一个为 ComfyUI 设计的综合图像处理插件，提供各种图像操作和增强功能。

## 🚀 功能特性

### 基础操作
- **ImageUpscale**: 使用多种插值方法缩放图片
- **ImageResize**: 将图片调整为精确尺寸
- **ImageRotate**: 按指定角度旋转图片
- **ImageFlip**: 水平和垂直翻转图片
- **ImageCrop**: 裁剪图片到指定区域

### 图片增强
- **ImageAdjustBrightness**: 修改图片亮度级别
- **ImageAdjustContrast**: 调整图片对比度
- **ImageBlur**: 应用高斯模糊
- **ImageSharpen**: 锐化图片细节
- **ImageGrayscale**: 将图片转换为灰度

### 智能缩放
- **ImageScaleByShortSide**: 保持宽高比的同时基于短边缩放
- **ImageScaleByLongSide**: 保持宽高比的同时基于长边缩放

## 📦 安装方法

1. 导航到您的 ComfyUI `custom_nodes` 目录
2. 克隆或复制此仓库：
   ```bash
   git clone https://github.com/zn123/ComfyUI-image-processor-zn123
   ```
3. 重启 ComfyUI
4. 节点将出现在 `image/processor` 分类中

## 🔧 使用方法

### 基础图片操作

#### ImageUpscale
使用各种插值方法缩放图片：
- **方法**: 最近邻、双线性、双三次、区域、Lanczos
- **缩放因子**: 图片尺寸的倍数
- **裁剪**: 可选的裁剪以保持宽高比

#### ImageResize
将图片调整为精确尺寸：
- **宽度**: 目标宽度（像素）
- **高度**: 目标高度（像素）
- **方法**: 调整大小的插值方法

#### ImageFlip
水平或垂直翻转图片：
- **水平**: 左右翻转（默认：True）
- **垂直**: 上下翻转

### 智能缩放

#### ImageScaleByShortSide
保持宽高比的同时基于短边缩放：
- **短边目标**: 短边的目标长度
- **放大方法**: 插值方法
- **仅放大**: 如果图片已经足够大则跳过处理

#### ImageScaleByLongSide
保持宽高比的同时基于长边缩放：
- **长边目标**: 长边的目标长度
- **放大方法**: 插值方法
- **仅放大**: 如果图片已经足够大则跳过处理

### 图片增强

#### ImageAdjustBrightness & ImageAdjustContrast
调整图片亮度和对比度级别：
- **亮度**: 范围（-1.0 到 1.0，0 = 无变化）
- **对比度**: 范围（-1.0 到 1.0，0 = 无变化）

#### ImageBlur & ImageSharpen
应用模糊或锐化效果：
- **模糊半径**: 高斯模糊半径（0.1 到 10.0）
- **锐化强度**: 锐化强度（0.0 到 2.0）

## 🎯 工作流程

### 基础图片预处理
1. 使用 `LoadImage` 节点加载图片
2. 使用 `ImageScaleByShortSide` 标准化图片尺寸
3. 应用 `ImageAdjustBrightness` 和 `ImageAdjustContrast` 进行曝光校正
4. 使用 `ImageSharpen` 增强细节
5. 输出到其他处理节点

### 智能调整大小工作流
1. 从任何来源输入图片
2. 使用 `ImageScaleByShortSide` 并设置 `only_upscale=True` 确保最小尺寸
3. 使用 `ImageCrop` 移除不需要的区域
4. 根据需要应用增强节点

## 📊 节点参考

| 节点名称 | 输入 | 输出 | 描述 |
|-----------|-------|--------|-------------|
| ImageUpscale | IMAGE, scale_by, method, crop | IMAGE | 按因子缩放图片 |
| ImageResize | IMAGE, width, height, method | IMAGE | 调整为精确尺寸 |
| ImageRotate | IMAGE, angle, method | IMAGE | 按度数旋转 |
| ImageFlip | IMAGE, flip_horizontal, flip_vertical | IMAGE | 翻转图片 |
| ImageCrop | IMAGE, x, y, width, height | IMAGE | 裁剪到区域 |
| ImageAdjustBrightness | IMAGE, brightness | IMAGE | 调整亮度 |
| ImageAdjustContrast | IMAGE, contrast | IMAGE | 调整对比度 |
| ImageBlur | IMAGE, radius | IMAGE | 应用高斯模糊 |
| ImageSharpen | IMAGE, amount | IMAGE | 锐化图片 |
| ImageGrayscale | IMAGE | IMAGE | 转换为灰度 |
| ImageScaleByShortSide | IMAGE, target, method, only_upscale | IMAGE, width, height | 按短边缩放 |
| ImageScaleByLongSide | IMAGE, target, method, only_upscale | IMAGE, width, height | 按长边缩放 |

## 🛠️ 技术细节

### 数据类型
- **输入**: ComfyUI `IMAGE` 张量格式（BHWC）
- **输出**: 相同格式的处理后数据
- **批处理**: 所有节点都支持批处理

### 插值方法
- **最近邻**: 最快，质量最低
- **双线性**: 速度和质量的良好平衡
- **双三次**: 更高质量，中等速度
- **区域**: 最适合缩小
- **Lanczos**: 最高质量，最慢

### 性能说明
- 智能缩放节点在尺寸匹配目标时跳过处理
- 批处理针对多张图片进行了优化
- 内存使用随图片尺寸扩展

## 🐛 故障排除

### 常见问题
1. **节点未出现**: 检查安装后是否重启了 ComfyUI
2. **内存错误**: 减少图片尺寸或批处理大小
3. **质量问题**: 尝试不同的插值方法

### 性能技巧
- 使用 `仅放大=True` 跳过不必要的处理
- 根据您的质量/速度需求选择插值方法
- 尽可能批量处理多张图片

## 🤝 贡献

1. Fork 仓库
2. 创建功能分支
3. 进行更改
4. 如适用，添加测试
5. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详情请参见 LICENSE 文件。

## 🙏 致谢

- ComfyUI 社区提供的框架
- 贡献者和测试者
- 开源图像处理库

## 📞 支持

- **问题**: 通过 GitHub Issues 报告错误
- **讨论**: 使用 GitHub Discussions 提问
- **社区**: 加入 ComfyUI Discord 服务器

---

**版本**: 1.0.0  
**最后更新**: 2024年  
**兼容性**: ComfyUI v0.3.15+