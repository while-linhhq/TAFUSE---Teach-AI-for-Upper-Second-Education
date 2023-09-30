# 1. Matplotlib:

## 1.1 Installation:

```terminal 
pip install matplotlib
```

### # Example

```python
import matplotlib

print(matplotlib.__version__)
```

## 1.2 Pylot Module:

```python
import matplotlib.pyplot as plt
import numpy as np

x_point = np.array([0, 6])
y_point = np.array([0, 250])

plt.plot(x_point, y_point, 'o')
plt.show()
```

- Nếu đoạn code trên không có đầy đủ 2 mảng tọa độ x, y thì chương trình sẽ tự động mặc đinh x lần lượt là 1, 2,3 ...
- Trong đoạn code trên 'o' để đánh dấu các điểm nối với nhau,có thể thay thế bằng ***marker = 'o'***

## 1.3 Marker in Pyplot:
- Thuộc tính ***marker*** có thể đi với các biến như sau:
  - ' o ' : circle
  - ' * ' : start
  - ' , ' : point
  - ' x ' : x
  - Có thể xem thêm tại: <https://www.w3schools.com/python/matplotlib_markers.asp/>
```python
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, marker = 'o')
plt.show()
```

- Ngoài ra còn có những thuộc tính khác:
  - Line:
    - ' - ' : Solid line
    - ' : ' : Dotted line
    - ' -- ' : Dashed line
    - ' -. ' : Dashed/Dotted line
  - Color:
    - ' r ' : Red
    - ' g ' : Green
    - ' b ' : Blue
    - Còn rất nhiều tham số color khác, hoặc có thể điền mã màu của nó thay vì kí tự có sẵn
- Các thuộc tính có thể viết tắt liền kề nhau theo thứ tự ***marker|line|color***, ví dụ: 'o:r'
 ```python
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, 'o:r')
plt.show()
```
## 1.4 Line:
- Bạn có thể định dạng kiểu **line** mà bạn muốn như dưới đây:

```python
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, linestyle = '-', color = '#4CAF50')
plt.show()
```
- Chèn nhiều ***line*** trong biểu đồ:

```python
import matplotlib.pyplot as plt
import numpy as np

x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 8, 1, 10])
x2 = np.array([0, 1, 2, 3])
y2 = np.array([6, 2, 7, 11])

plt.plot(x1, y1, x2, y2)
plt.show()
```

## 1.5 Label:
- Có thể chèn thông tin các cột và tên của biểu đồ bằng cách dưới đây:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.show()
```

## 1.6 Grid:
- Tiếp tục với ví dụ trên khi muốn vẽ ***Grid*** ta có thể:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)

plt.grid(axis = 'y', color = 'green', linestyle = '--', linewidth = 0.5)

plt.show()
```

## 1.7 Multiple Plot:
- Cho kết quả đầu ra nhiều biểu đồ cũng lúc:

```python
import matplotlib.pyplot as plt
import numpy as np

#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(1, 2, 1)
plt.plot(x,y)

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(1, 2, 2)
plt.plot(x,y)

plt.show()
```
- ***plt.subplot()*** có 3 tham số lần lượt bao gồm: số dòng, số cột, vị trí biểu đồ hiển thị.
- Cùng là đoạn code trên nếu thay đổi số cột và dòng lần lượt là 2, 1 thì nó sẽ hiển thị theo hình thức trình bày khác.
- Thêm vào đó, ta có thể thêm tên của từng biểu đồ như sau:

```python

import matplotlib.pyplot as plt
import numpy as np

#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(1, 2, 1)
plt.plot(x,y)
plt.title("SALES")

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(1, 2, 2)
plt.plot(x,y)
plt.title("INCOME")

plt.show()
```
- Tuy nhiên cách trên mới chỉ hiển thị tên của từng biểu đồ, còn nếu bạn muốn đặt tên chung cho tất cả các biểu đồ thì sao? Tham khảo cách dưới đây nhé

```python
import matplotlib.pyplot as plt
import numpy as np

#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(1, 2, 1)
plt.plot(x,y)
plt.title("SALES")

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(1, 2, 2)
plt.plot(x,y)
plt.title("INCOME")

plt.suptitle("MY SHOP")
plt.show()
```
- Trong đó hàm ***suptitle()*** đóng vai trò hiển thị tên chung cho biểu đồ, hàm ***title()*** hiển thị tên riêng của từng biểu đồ.

## 1.8 Scatter:

- Ta có ví dụ sau: 

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
plt.scatter(x, y, color = 'hotpink')

x = np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
plt.scatter(x, y, color = '#88c999')

plt.show()
```
## 1.9 Bar Chart:
- Còn nếu muốn hiển thị biểu đồ cột, ta có thể áp dụng như sau:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.bar(x,y)
plt.show()
```
- Nếu muốn biểu đồ cột quay ngang thì ta thay hàm ***.bar()*** thành ***.barh()***

## 1.10 Pie Chart:
- Biểu đồ tròn:

```python
import matplotlib.pyplot as plt
import numpy as np

y = np.array([35, 25, 25, 15])
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
myexplode = [0.2, 0, 0, 0]

plt.pie(y, labels = mylabels, startangle = 90, explode=myexplode, shadow= True)
plt.legend(title = "Four Fruits:")
plt.show()
```
## 1.11 Histogram:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.random.normal(170, 10, 250) ##standard deviation

plt.hist(x)
plt.show()
```

# 2. Exercise:

*Tận dụng kết hợp tất cả các module trên thành 1 bài hoàn chỉnh*
