# 1 PyTorch 모델을 REST API로 배포하기

> [파이토치 한국 사용자 모임: REST API 배포하기](https://tutorials.pytorch.kr/intermediate/flask_rest_api_tutorial.html)

이 문서에서는 (1) Flask를 사용해서 PyTorch model을 배포하고, (2) model inference를 할 수 있는 REST API를 만들어 볼 것이다.

---

## 1.1 API 정의

먼저 API endpoint의 request(요청)와 response(응답)을 정의할 것이다.

- request: 이미지가 포함된 `file` parameter를 HTTP POST로 `/predict`에 요청한다.

- response: JSON 형태로 다음과 같은 predict 결과를 포함한다.

```json
{"class_id": "n02124075", "class_name": "Egyptian_cat"}
```

> API endpoint란 어느 application이 다른 application에 서비스를 요청하는 방식이다.

---

## 1.2 dependencies 설치

다음 명령으로 필요한 package를 설치한다.

```bash
$ pip install Flask==2.0.1 torchvision==0.10.0
```

---

## 1.3 간단한 Web server 구성

우선 아래 코드를 `app.py`라는 파일명으로 저장한 뒤 Flask 개발 server를 실행할 것이다.

```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'
```

`app.py` 작성이 끝났다면 다음 명령으로 Flask 개발 server를 실행할 수 있다.

```bash
$ FLASK_ENV=development FLASK_APP=app.py flask run
```

이제 웹 브라우저로 http://localhost:5000/에 접속하면 app.py에 작성한 "Hello, World"가 표시된다.

이제 API 정의에 맞게 `app.py`를 수정해 보자.

- method 이름을 `hello()`에서 `predict()`로 변경한다.

- endpoint path(경로)를 `/predict`로 변경한다.

- image 파일은 HTTP POST request로 보내지기 때문에, POST 요청에만 허용하도록 만든다.

```python
@app.route('/predict', methods=['POST'])
def predict():
    return 'Hello, World!'
```

그 다음은 회신 받을 JSON의 형태로 응답 형식을 변경해 보자.

- ImageNet classification `class_ID`, `class_name`

```python
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})
```

---

## 1.4 inference

다음은 model inference를 진행하는 코드이다. 코드를 살펴본 뒤 어떻게 Web server와 통신하는지 알아보자.

1. transform pipeline 구축

DenseNet model이 (224, 224, 3) input image를 사용하기 때문에 다음과 같은 transform pipeline을 구축한다.

- 동시에 image tensor를 mean, std 값으로 Normalize한다.

- image를 byte 단위로 읽은 뒤, 정의한 transform 과정을 거쳐서 Tensor를 반환한다.

```python
import io

import torchvision.transforms as transforms
from PIL import Image

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)
```

위 pipeline 코드가 제대로 작동하는지 테스트해 보자.

```python
# 경로는 본인의 환경에 맞게 수정
with open("../_static/img/sample_file.jpeg", 'rb') as f:
    image_bytes = f.read()
    tensor = transform_image(image_bytes=image_bytes)
    print(tensor)
```

2. model load, prediction

이제 model을 불러와서 classification을 진행해 보자. pretrained DenseNet 121 model을 사용할 것이다.

> model을 **global variable**로 둬서 request마다 model을 불러오지 않도록 한다.

```python
from torchvision import models

# 이미 학습된 가중치를 사용하기 위해 `pretrained` 에 `True` 값을 전달합니다:
model = models.densenet121(pretrained=True)
# 모델을 추론에만 사용할 것이므로, `eval` 모드로 변경합니다:
model.eval()


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat
```

결과로 반환하는 `y_hat` Tensor는 predict `class_ID`와 `index`를 포함한다.

- `index`를 사람이 읽을 수 있는 `class_name`으로 변환하기 위해, `imagenet_class_index.json` 파일을 사용한다.

  > [imagenet_class_index.json](https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json)

다음은 앞서 본 prediction 코드에서, `imagenet_class_index.json`을 읽고 `y_hat`의 `index`를 `class_name`으로 변환하도록 고친 코드이다.

```python
import json

imagenet_class_index = json.load(open('../_static/imagenet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)

    # 예측값(y_hat) index를 class_name으로 변환
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]
```

> json의 key(index)가 `string` type이기 때문에, `str(y_hat.item())`로 변환한 것에 유의하자.

이제 위 `get_prediction()` method에 input으로 image 하나를 줘서 test해 보자.

```Python
with open("../_static/img/sample_file.jpeg", 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))
```

결과는 다음과 같다. 각각 예측값의 `class_id`(ImageNet 분류 ID), `class_name`(사람이 읽을 수 있는 이름)을 의미한다.

```
['n02124075', 'Egyptian_cat']
```

---

## 1.5 model을 API server와 통합하기

이제 model inference을 수행해 주는 Flask API server에 model을 추가해 줄 것이다.

- API server는 image file을 받으므로, `request` 모듈이 `file`을 읽도록 `predict()` method 부분을 수정한다.

  > 가령 JSON 형식으로 전달된 request는 `request.get_json()`으로 읽을 수 있다.

```python
from flask import request

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # image file을 받으므로 아래와 같이 작성한다.
        file = request.files['file']
        # 받은 byte stream을 읽어서 img_bytes 변수에 저장한다.
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})    
```

---

## 1.6 전체 코드

> JSON path(<PATH/TO/.json/FILE>)를 본인 환경에 맞게 수정한다.

```python
import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request


app = Flask(__name__)
imagenet_class_index = json.load(open('<PATH/TO/.json/FILE>/imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()
```

마찬가지로 다음 명령을 이용해 server를 실행한다.

```bash
$ FLASK_ENV=development FLASK_APP=app.py flask run
```

---

## 1.7 API server test

`request` library를 사용해서 **POST request**를 한 번 보내서 test해 보자.

> JSON path(<PATH/TO/.json/FILE>)를 본인 환경에 맞게 수정한다.

```python
import requests

# test용 POST request 
resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('<PATH/TO/.jpg/FILE>/cat.jpg','rb')})

# 요청을 보낸다.
resp.json()
```

**HTTP response** 결과는 다음과 같다.

```json
{"class_id": "n02124075", "class_name": "Egyptian_cat"}
```

---

### 1.7.1 curl 이용 예시

`curl`을 이용해서도 API로 image를 전달할 수 있다. dog.jpg를 전달할 것이다.

- `-X POST`: POST request를 수행하는 명령.

- `-F image=@dog.jpg`: image(dog.jpg)를 제공하는 명령.

```bash
$ curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
{
  "predictions": [
    {
      "label": "beagle",
      "probability": 0.9901360869407654
    },
    {
      "label": "Walker_hound",
      "probability": 0.002396771451458335
    },
    {
      "label": "pot",
      "probability": 0.0013951235450804234
    },
    {
      "label": "Brittany_spaniel",
      "probability": 0.001283277408219874
    },
    {
      "label": "bluetick",
      "probability": 0.0010894243605434895
    }
  ],
  "success": true
}
```

---