# Basic Machine Learning

## 1. Introduction:
- Trong nội dung ***Basic Machine Learning*** này chúng ta sẽ tìm hiểu đại khái về cách đào tạo máy học cơ bản là như thế nào, biết thêm về tầm quan trọng của toán học đối với nó và cũng học thêm những **function** để dự đoán kết quả từ những gì máy tính đã học được.
  - Trong suy nghĩ của máy tính kiến thức nó có thể học là một bảng dữ liệu có thể là tập hợp của nhiều kiểu dữ liệu, miễn là nó logic và là một khối thống nhất
  - Bạn có thể tham khảo 1 ví dụ tập hợp dữ liệu ở đây: <https://www.w3schools.com/python/python_ml_getting_started.asp/>
  - Tuy nhiên không phải thông tin dữ liệu nào ta cũng lấy để đào tạo máy học như bảng trên, cụ thể hơn với trường **Color** nó sẽ không giúp ích gì đến việc dự đoán thông số **AutoPass** hoặc **Speed**. Vì vậy hãy cân nhắc trước khi sử dụng một dataset có sẵn hay để ý đến những gì bản thân cần để giải một bài toán nhé.
  - Chung chung lại, là kĩ năng phân tích dữ liệu phải chắc chắn.
  - Sẽ có 3 kiểu dữ liệu chính sau đây: ***số liệu, kí tự, tổng hợp***
    - Số liệu: là kiểu dữ liệu có thể đo lường và được chia làm 2 loại là số nguyên và số thực. Ví dụ số lượng ô tô đi ngang qua hay tỉ lệ lượng đường trong máu
    - Kí tự: sẽ trả về true/false hoặc kí tự tượng trưng nào đó, ví dụ như tên bệnh mắc phải
    - Tổng hợp: Là sự pha trộn của số liệu và kí tự, ví dụ như: Linh cao hơn Tú 5cm
  
## 2. Định nghĩa Mean, Median và Mode:
- Đánh giá mức độ, xu hướng tập trung của dữ liệu dựa trên 3 tham số đó.
- Mean: trả về giá trị trung bình của một mảng

  - Ví dụ: **(99+86+87+88+111+86+103+87+94+78+77+85+86) / 13 = 89.77**
```python
import numpy

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = numpy.mean(speed)

print(x)
```

- Media: trả về giá trị trung vị trong tập hợp mảng
  - Là dạng con của dạng tổng quát có tên là Tứ phân vị
```python
import numpy

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = numpy.median(speed)

print(x)
```
- Mode: trả về giá trị xuất hiện nhiều nhất trong mảng:

```python
from scipy import stats

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = stats.mode(speed)

print(x)
```
## 3. Phương sai và độ lệch chuẩn:
- Đo lường sự biến thiên, phân tán của dữ liệu
- Kiểm soát tốt 2 khai niệm trên ta cũng có thể kiểm soát những ảnh hưởng sau:
  - Overfitting
  - Noise reduction
  - Build model
### 3.1 Phương sai:
- Giải thích thêm: Trong xác suất thống kê có 2 khái niệm cơ bản là ***population*** và ***sample***. Population là tập hợp có số lượng lớn các cá thể và một sample là một tập hợp con hay tập mẫu của population.

- Công thức: Sẽ có 2 công thức khá tương tự nhau khi tính phương sai:
  - Đối với sample: ![population](https://cldup.com/VErEhbqy5K-3000x3000.png)
  - Đối với population: ![sample](https://cldup.com/B_DCgio2ID-3000x3000.png)
- Lý do khi tính phương sai của mẫu dữ liệu ta phải dùng "n-1", để đảm bảo tính chính xác và đúng đắn trong việc ước tính phương sai tổng thể từ mẫu con, tránh sai lệch và cho biểu đồ chính xác khi đưa ra kết luận tổng thể dựa trên mẫu, không phải là sự thiếu sót hay lãng phí.(***Degrees of Freedom***)
```python
import numpy

speed = [32,111,138,28,59,77,97]

x = numpy.var(speed)

print(x)
```
### 3.2 Độ lệch chuẩn:
- Giải thích: Cũng tương tự phương sai, độ lệch chuẩn dùng để đo lường sự biến đổi của dữ liệu cũng như mức độ phân tán của nó. Ngoài ra nó cũng thường được sử dụng để tính toán khoảng tin cậy trong thống kê.
- Công thức: ![standard_deviation](https://cldup.com/xYiN6A8o7X-3000x3000.png)
```python
import numpy

speed = [32,111,138,28,59,77,97]

x = numpy.std(speed)

print(x)
```
### 3.3 Tính tương quan:
- Trong xác suất thống kê, hệ số tương quan cho biết độ mạnh của mối quan hệ tuyến tính giữa 2 biến số ngẫu nhiên. Nếu phương sai đo lường sự biến thiên của một biến ngẫu nhiên(hay dữ liệu trên một tập mẫu) thì hiệp phương sai đo lường sự biến thiên của hai biến ngẫu nhiên.
- Công thức:
  - Hiệp phương sai: ![population](https://cldup.com/BU7VQs5VdH-3000x3000.png)
  - Hệ số tương quan: ![population](https://cldup.com/03fGbeVD2F-3000x3000.png)
```python
import matplotlib.pyplot as plt
import numpy as np
Temperature = [14.2, 16.4,11.9, 15.2, 18.5, 22.1, 19.4, 25.1, 23.4, 18.1, 22.6, 17.2]
 
Ice_Cream_Sales = [215, 325, 185, 332, 406, 522, 412, 614, 544, 421, 445, 408]
 
plt.scatter(Temperature,Ice_Cream_Sales)
 
plt.show()
print(np.corrcoef(Temperature, Ice_Cream_Sales)) # 0.9575
```
## 4. Percentile:
- Là hàm thuộc thư viện numpy với mục đích tìm tỉ lệ phần trăm của những giá trị thấp hơn 1 giá trị nào đấy trong mảng.

```python
import numpy

ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]

x = numpy.percentile(ages, 75)

print(x)
```
## 5. Phân phát dữ liệu:
- Phần phát dữ liệu với tính bất kỳ:
```python
import numpy
import matplotlib.pyplot as plt

x = numpy.random.uniform(0.0, 5.0, 100000)

plt.hist(x, 100)
plt.show()
```

- Phân phát dữ liệu xoay quanh một giá trị nào đó:
```python
import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 100000)

plt.hist(x, 100)
plt.show()
```
- Biểu diễn một tệp dữ liệu trên dạng sơ đồ điểm:
```python
import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 1000)
y = numpy.random.normal(10.0, 2.0, 1000)

plt.scatter(x, y)
plt.show()
```
## 6. Hồi quy tuyến tính:
- Phương pháp này dùng để dự đoán và tìm mối quan hệ giữa các điểm dữ liệu để vẽ một đường thẳng hồi quy tuyến tính.
- Dưới đây sẽ là dữ liệu mà ta sẽ phân tích và áp dụng trong bài này (x là đại diện tuổi xe và y đại diện cho tốc độ tối đa của xe, coi các thông số và yếu tố khác đều giống nhau)

```python
import matplotlib.pyplot as plt

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x, y)
plt.show()
```
- Để vẽ đường hồi quy tuyến tính ta sử dụng thư viện Scipy:

```python
import matplotlib.pyplot as plt
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
```


- Dự đoán kết quả ta làm như sau:

```python
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

speed = myfunc(10)

print(speed)
```

## 7. Hồi quy đa thức:
- Nếu các điểm dữ liệu rõ ràng không phù hợp với Hồi quy tuyến tính thì đây sẽ là một lựa chọn lý tưởng của bạn
- Dữ liệu dưới đây chỉ ra với từng giờ cụ thể trong ngày thì tốc độ trung bình của xe đi qua sẽ là bao nhiêu.

```python
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

plt.scatter(x, y)
plt.show()
```
- Vẽ parabol ta thu được biểu đồ sau khi chạy code:

```python
import numpy
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()
```
- Giải thích:
  - Numpy có một phương thức tạo mô hình đa thức (dòng 5)
  - Quy định giới hạn vẽ từ 1 đến 22 và lấy 100 mẫu (Dòng 6)

## 8. Hồi quy bội:
- Tương tự như hồi quy tuyến tính nhưng nó có thể dự đoán 1 giá trị dựa trên nhiều giá trị đầu vào
- Chẳng hạn như ta sử dụng dataset ở link sau: <https://www.w3schools.com/python/python_ml_multiple_regression.asp/>

```python
import pandas
from sklearn import linear_model

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)
print(regr.coef_)
```

- Ta sử dụng thư viện **sklearn** để tạo mô hình hồi quy bội
- ***regr.coef_*** là hệ số miêu tả mối quan hệ giữa kết quả đầu ra và dữ liệu đầu vào:
  - [0.00755095 0.00780526], nó miêu tả mỗi khối lượng tăng 1kg thì lượng Co2 tăng tương đương với 0.00755095 và tương tự đối với thể tích xi lanh.

## 9. Tỉ lệ:
- Có thể tham khảo dataset được sử dụng ở đây: <https://www.w3schools.com/python/python_ml_multiple_regression.asp/>
- Khi sử dụng 2 kiểu dữ liệu khác nhau. Ví dụ: kg và m, độ cao và thời gian. Lúc này ta phải tiêu chuẩn hóa dữ liệu đầu vào đó để thuận tiện trong việc so sánh và sử dụng. Ngoài ra, lúc này ta sẽ không cần quan tâm nó thuộc kiểu dữ liệu hay đơn vị nào nữa vì sau khi tiêu chuẩn hóa thì các dữ liệu đầu vào có đơn vị như nhau.
- Công thức: ***z = (x - u) / s***
  - ***z*** là giá trị mới
  - ***x*** là giá trị nguyên bản
  - ***u*** là giá trị mean
  - ***s*** là độ lệch chuẩn
- Ví dụ đối với dữ liệu trên:
  - ***weight***: (790 - 1292.23) / 238.74 = -2.1
  - ***volume***: (1.0 - 1.61) / 0.38 = -1.59
  - Lúc này ta so sánh -2.1 và -1.59 giống như so sánh giữa 790 và 1.0
- Trong python sklearn có modules ***StandardScaler()*** để trả về giá trị sau khi tiêu chuẩn hóa

```python
import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)
```
## 10. Train/Test
- Ta tự tạo dataset như sau:

```python
import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

plt.scatter(train_x, train_y)
plt.show()
```

- Cách chia tỉ lệ tệp test và tệp train thông thường là 80 | 20


```python
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(test_y, mymodel(test_x))

print(r2)
print(mymodel(5))

myline = numpy.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()
```

## 11. Cây quyết định:

- Miêu tả cho việc quyết định 1 thứ gì đó dựa trên những kinh nghiệm trước đó
- Chẳng hạn ta có dataset sau: <https://www.w3schools.com/python/python_ml_decision_tree.asp/>

```python
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
#import matplotlib.pyplot as plt

df = pandas.read_csv("data.csv")

d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

print(dtree.predict([40, 10, 7, 1]))

tree.plot_tree(dtree, feature_names=features)
```

- Giải thích: 
  - Ta phải định dạng lại kiểu dữ liệu về cùng 1 kiểu dữ liệu nhất định, sử dụng phương thức ***map()***