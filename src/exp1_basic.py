"""
实验1：CKKS图像加密基础验证
- 读取灰度图像
- CKKS加密
- 测量加密/解密时间
- 计算密文膨胀率
- 执行亮度增强（加常数）
- 计算PSNR评估精度
"""

import tenseal as ts
import numpy as np
from PIL import Image
import time
import os
import matplotlib.pyplot as plt

# ==================== 参数设置 ====================
image_path = "/home/niuxiyao/fhe_image/images/mandril_gray.tif"
# CKKS参数
poly_modulus_degree = 8192      # 多项式模数（越大安全性越高，但速度越慢）
coeff_mod_bit_sizes = [60, 40, 40, 60]  # 系数模数链
global_scale = 2**40             # 缩放因子（影响精度和效率的平衡）

# ==================== 1. 创建CKKS上下文 ====================
print("正在创建CKKS上下文...")
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree,
    coeff_mod_bit_sizes
)
context.global_scale = global_scale
# 生成加解密密钥（自动包含）
# 如果需要计算（密文运算），必须生成Galois密钥和重线性化密钥
context.generate_galois_keys()
context.generate_relin_keys()
print("上下文创建完成。")

# ==================== 2. 加载并预处理图像 ====================
print("\n正在加载图像...")
img = Image.open(image_path).convert('L')  # 强制转为灰度图
img_array = np.array(img, dtype=np.float32) / 255.0  # 归一化到[0,1]
height, width = img_array.shape
img_flat = img_array.flatten().tolist()
print(f"图像尺寸: {width} x {height}, 像素总数: {len(img_flat)}")

# ==================== 3. 加密图像 ====================
print("\n正在加密图像...")
start_encrypt = time.time()
encrypted_img = ts.ckks_vector(context, img_flat)  # 加密
encrypt_time = time.time() - start_encrypt
print(f"加密耗时: {encrypt_time:.4f} 秒")

# 获取密文大小（粗略估计，序列化后的字节数）
encrypted_bytes = encrypted_img.serialize()  # 序列化为字节
print(f"密文大小: {len(encrypted_bytes) / 1024:.2f} KB")
print(f"原始图像大小: {img_array.nbytes / 1024:.2f} KB")
print(f"密文膨胀率: {len(encrypted_bytes) / img_array.nbytes:.2f} 倍")

# ==================== 4. 解密并验证正确性 ====================
print("\n正在解密...")
start_decrypt = time.time()
decrypted_flat = encrypted_img.decrypt()  # 解密
decrypt_time = time.time() - start_decrypt
print(f"解密耗时: {decrypt_time:.4f} 秒")

# 重构图像
decrypted_array = np.array(decrypted_flat).reshape(height, width)
# 计算重建误差（MSE 和 PSNR）
mse = np.mean((decrypted_array - img_array) ** 2)
if mse == 0:
    psnr = float('inf')
else:
    psnr = 10 * np.log10(1.0 / mse)  # 像素值范围[0,1]
print(f"解密图像与原始图像的 MSE: {mse:.6f}")
print(f"PSNR: {psnr:.2f} dB")

# ==================== 5. 对加密图像进行亮度增强 ====================
print("\n正在执行加密域亮度增强...")
brightness_factor = 0.2  # 增加0.2的亮度
# 在加密域中加一个常数（注意要构造相同长度的常数向量）
constant_vec = [brightness_factor] * len(img_flat)
encrypted_bright = encrypted_img + constant_vec

# 解密增强后的图像
bright_decrypted_flat = encrypted_bright.decrypt()
bright_array = np.array(bright_decrypted_flat).reshape(height, width)

# 明文亮度增强（作为参考）
plain_bright = img_array + brightness_factor
plain_bright = np.clip(plain_bright, 0, 1)  # 截断到有效范围

# 计算解密增强图像与明文增强图像的差异
mse_bright = np.mean((bright_array - plain_bright) ** 2)
psnr_bright = 10 * np.log10(1.0 / mse_bright) if mse_bright > 0 else float('inf')
print(f"加密域亮度增强结果与明文结果的 MSE: {mse_bright:.6f}")
print(f"PSNR: {psnr_bright:.2f} dB")

# ==================== 6. 保存结果图像（用于论文插图） ====================
# 创建结果保存目录（如果不存在）
results_dir = "/home/niuxiyao/fhe_image/results/exp1"
os.makedirs(results_dir, exist_ok=True)

# 将图像保存为PNG（注意像素值还原到0-255）
def save_image(array, filename):
    img_save = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
    img_save.save(os.path.join(results_dir, filename))

save_image(img_array, "original.png")
save_image(decrypted_array, "decrypted.png")
save_image(plain_bright, "plain_bright.png")
save_image(bright_array, "encrypted_bright.png")

# 绘制对比图
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original')
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(decrypted_array, cmap='gray')
plt.title(f'Decrypted (PSNR: {psnr:.2f} dB)')
plt.axis('off')
plt.subplot(2, 3, 3)
# 显示误差图（放大差异）
error_map = np.abs(decrypted_array - img_array)
plt.imshow(error_map, cmap='hot', vmin=0, vmax=0.05)
plt.title('Error Map')
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(plain_bright, cmap='gray')
plt.title('Plain Brightness')
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(bright_array, cmap='gray')
plt.title(f'Encrypted Bright (PSNR: {psnr_bright:.2f} dB)')
plt.axis('off')
plt.subplot(2, 3, 6)
error_bright = np.abs(bright_array - plain_bright)
plt.imshow(error_bright, cmap='hot', vmin=0, vmax=0.05)
plt.title('Brightness Error')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "comparison.png"), dpi=150)
plt.show()

print(f"\n实验结果已保存到: {results_dir}")