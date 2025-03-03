from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')

# 使用 PaddleOCR 繁體中文模型
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  

# OCR 辨識
image_path = "Your Image Path"

chinese_tra = plt.imread(image_path)

fig, ax = plt.subplots()
ax.imshow(chinese_tra)

result = ocr.ocr(image_path, cls=True)

# 顯示結果
for line in result:
    for word_info in line:
        box = word_info[0]
        ax.plot([point[0] for point in box] + [box[0][0]], 
                [point[1] for point in box] + [box[0][1]], 
                color='red', linewidth=2)
        text_x,text_y = box[0][0], box[0][1]
        text, confidence = word_info[1]
        print(f"辨識文字: {text}, 置信度: {confidence:.2f}")
        ax.text(box[0][0], box[0][1], text, fontsize=10, color='red')
        ax.text(box[1][0], box[0][1], f"{confidence:.2f}", fontsize=10, color='red')

# plt.imsave('plot_file.jpg', ax)
plt.show()