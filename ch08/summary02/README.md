# 8 텐서플로 서빙을 사용한 모델 배포

## 8.5 TensorFlow Serving 설정

model 내보내기와 signature 작성, test까지 모두 마쳤으므로, 이제 TensorFlow Serving을 위한 server 구축 작업을 살펴보자.

TF에서 제공하는 docker image를 사용하거나, Ubuntu에서 TF serving용 package를 설치하는 방법이 있다.

---

### 8.5.1 Docker를 이용한 Serving

docker image는 다음 명령으로 설치할 수 있다.

```bash
$ docker pull tensorflow/serving
```

만약 NVIDIA GPU를 이용하여 Docker Container를 실행한다면 다음과 같이 GPU를 지원하는 최신 빌드로 설치한다.

```bash
$ docker pull tensorflow/serving:latest-gpu
```

---

### 8.5.2 Ubuntu package를 이용한 Serving

Ubuntu package를 이용하는 방법은 다음과 같다. 우선 package source를 설치한다.

```bash
$ echo "deb [arch=arm64] http://storage.googleapis.com/tensorflow-serving-apt \
stable tensorflow-model-server tensorflow-model-server-universal" \
| sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
```

설치가 끝났다면 update를 수행하기 전에 먼저 배포 키 체인에 public key를 추가해야 한다.

```bash
$ curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
```

이제 update를 수행하면 TensorFlow Serving 설치가 끝난다.

```bash
$ apt-get update
$ apt-get install tensorflow-model-server
```

---

## 8.6 TensorFlow Server 구성

TF Serving은 두 가지 모드로 실행할 수 있다.

- 지정한 model을 load하는 모드.(항상 최신 model을 사용하게 설정할 수 있다.)

  > 앞서 model을 내보낼 때 timestamp가 자동으로 추가되었다. 이러한 timestamp를 버전 번호로 사용해서 최신 model을 구분할 수 있는 것이다.

  > file_system_poll_wait_seconds 옵션이 0보다 클 때만 자동으로 작동한다.(default는 2s)

- 구성 파일(model, version 정보를 포함)을 지정한 뒤, 상황에 따라 필요한 model을 load하는 모드

---

### 8.6.1 단일 model 구성

우선 단일 model을 load하면서 항상 최신 version을 사용하는 예제를 살펴보자. 

1. Docker를 이용하는 방법

```bash
$ docker run -p 8500:8500 \ # 기본 port를 지정
             -p 8501:8501 \
             --mount type=bind,source=/tmp/models,target=/models/my_model \ # model directory를 mount
             -e MODEL_NAME=my_model \ # model을 지정(환경변수)
             -e MODEL_NAME_PATH=/models/my_model \
             -t tensorflow/serving # Docker image를 지정
```

이때 port를 8500, 8501 둘로 지정했는데, 이는 TF Serving이 REST 및 gRPC(Google Remote Procedure Call) endpoint를 모두 생성하도록 구성됐기 때문이다.

주의할 점은 앞서 GPU image용으로 docker image를 받았다면 마지막 `-t` 부분을 tensorflow/serving:latest-gpu로 전달해야 한다.

> 참고로 `--use_tflie_model=true` 옵션을 추가하면, server의 TFLite model도 load할 수 있다.(9장 참조)

2. Ubuntu package를 사용하는 방법

```bash
tensorflow_model_server --port=8500 \
                        --rest_api_port=8501 \
                        --model_name=my_model \
                        --model_base_path=/tmp/models/my_model
```

두 방식 모두 유사한 출력을 보여준다. my_model을 성공적으로 load했으며, 두 가지 endpoint(REST, gRPC)를 생성했다는 출력이다.

디렉터리에 새 model을 upload하면 모델 관리자가 새 버전을 알아서 검색하고 기존 model을 대체한다. 가령 docker의 mount된 폴더에 새 model을 추가하면, 별다른 구성 변경 없이 알아서 새 model을 검색한 뒤 endpoint를 다시 load하고 출력으로 표시한다.

> 또한 model rollback 기능도 지원하는데, 단순히 최신 버전을 디렉터리에서 삭제하면 된다. 그러면 알아서 삭제를 감지하고 이전 버전 중 최신 model을 load한다.

---

### 8.6.2 다중 model 구성

동시에 여러 model을 load하게 만들기 위해서는 configuration file을 만들어야 한다.

> model_platform 부분을 보면 알겠지만, TF Serving에서는 다양한 framework로 만들어진 model을 load할 수 있다. TensorFlow, MXNet, ONNX, Scikit-learn, XGBoost, PMML, H2o 등

- `model_config_list`라는 key 아래 `config` 딕셔너리들이 포함된다. 

```
model_config_list {
    config {
        name: 'my_model'
        base_path: '/models/my_model/'
        model_platform: 'tensorflow'
    }
    config {
        name: 'another_model'
        base_path: '/models/another_model/'
        model_platform: 'tensorflow'
    }
}
```

이때 특정 model version들만 load하고 싶다면 `config` 딕셔너리 내부에 다음과 같이 `model_version_policy` 쌍을 작성해 주면 된다.

```
...
    # 사용할 수 있는 모든 version을 load
    config {
        name: 'another_model'
        base_path: '/models/another_model/'
        model_version_policy: {all: {}}
    }
```

```
    # 특정 version만 load
    config {
        name: 'another_model'
        base_path: '/models/another_model/'
        model_version_policy: {
            specific {
                versions: 1556250435
                versions: 1556251435
            }
        }
    }
```

특정 version에 label을 붙여서 관리할 수도 있다.

```
    # 특정 version만 load
    config {
        name: 'another_model'
        base_path: '/models/another_model/'
        model_version_policy: {
            specific {
                versions: 1556250435
                versions: 1556251435
            }
        }
        version_labels {
            key: 'stable'
            value: 1556250435
        }
        version_labels {
            key: 'test'
            value: 1556251435
        }
    }
```

1. Docker를 이용하는 방법

configuration file을 mount한 부분과 `--model_config_file` 옵션으로 해당 파일을 지정한 것을 주목하자.

```bash
$ docker run -p 8500:8500 \
             -p 8501:8501 \
             --mount type=bind,source=/tmp/models,target=/models/my_model \
             --mount type=bind,source=/tmp/model_config,target=/models/model_config \
             -e MODEL_NAME=my_model \
             -t tensorflow/serving \
             --model_config_file=/models/model_config
```

2. Ubuntu package를 사용하는 방법

단순히 `--model_config_file` 옵션을 추가한 뒤 configuration file을 넘겨주면 된다.

```bash
tensorflow_model_server --port=8500 \
                        --rest_api_port=8501 \
                        --model_config_file=/models/model_config
```

---

## 8.7 REST vs gRPC

앞서 TF Serving은 REST와 gRPC 두 가지 API를 지원하는 것을 확인했다. 두 protocol 모두 HTTP를 기반으로 하지만 각자 장단점을 가진다.

1. REST

client가 web service와 통신하는 방법을 정의하는 protocol이다. 

- 표준 HTTP method(GET, POST, DELETE 등)를 사용하여 server와 통신한다.

- request payload(server에 전송되는 data)는 XML이나 JSON 형식으로 encoding하는 경우가 많다.

- endpoint를 `curl` 요청이나 브라우저 도구로 쉽게 테스트할 수 있다.

2. gRPC

구글에서 개발한 원격 프로시저 protocol이다. 표준 데이터 형식으로 protocol buffer를 사용한다. 덕분에 latency가 적은 빠른 통신이 가능하며 REST에 비해 더 적은 데이터로 전송할 수 있다.

> 하지만 binary 형식이기 때문에 검사가 어려울 수 있다.


---

## 8.8 model server inference: REST API

REST API로 model server에 request하기 위해서는 'requests'라는 python library를 설치해야 한다.

```bash
$ pip install requests
```

설치가 끝났으면 POST request를 보낼 수 있다. 간단한 예시를 보자.

```python
import requests

url = "http://some-domain.abc"
payload = {"key_1": "value_1"}
r = requests.post(url, json=payload) # request 제출
print(r.json()) # HTTP response 출력
```

---

### 8.8.1 URL 작성

지금 예제와 같이 HTTP request를 수행하기 위해서는 URL을 작성하는 형식을 알아둘 필요가 있다.

```bash
http://{HOST}:{PORT}/v1/models/{MODEL_NAME}:{VERB}
```

- HOST: model server의 IP address 또는 domain name

  > server와 client가 동일한 system이라면 localhost로 설정해도 된다.

- PORT: TF Serving에서 REST API의 표준 port인 8501를 넘기면 된다.(다른 서비스와 충돌하면 변경도 가능하다.)

- MODEL_NAME: model server에 load된 model의 이름과 일치해야 한다.

- VERB: signature 방식에 따라 적는다.(predict, classify, regress)

만약 특정 model version을 사용하고 싶다면 {MODEL_NAME} 뒤부터 URL을 확장해서 작성하면 된다.

```bash
http://{HOST}:{PORT}/v1/models/{MODEL_NAME}/versions/{MODEL_VERSION}:{VERB}
```

---

### 8.8.2 request payload 작성

request payload는 주로 JSON 형식으로 작성한다. 가장 기본적인 형태는 다음과 같다.

```
{
    "signature_name": <string>,
    "instances": <value>
}
```

> `signature_name`은 꼭 작성해 줄 필요는 없다. 지정하지 않으면 model server에서 default signature를 사용한다.

여러 data sample을 제출하고 싶다면, 다음과 같이 `instances` key 아래 list들로 제출할 수 있다.

```
{
    "instances": [
        {
            "input_1": [1, 2, 3, 4],
            "input_2": [5, 6, 7, 8]
        },
        {
            "input_1": [9, 10, 11, 12],
            "input_2": [13, 14, 15, 16]
        }
    ]
}
```

만약 단 하나의 input만 request하고 싶다면 `instances` 대신 `inputs`만 작성하면 된다. 단, 이 둘을 동시에 혼용해서 쓰면 안 되므로 주의하자.

```
{
    "inputs": [1, 2, 3, 4]
}
```

혹은 request 과정을 함수화하면 매개변수로 input text를 받아서, 여러차례 request하도록 구성할 수 있다.

```Python
import requests

def get_rest_request(text, model_name="my_model"):
    url = "http://localhost:8501/v1/models/{}:predict".format(model_name)
    payload = {"instances": [text]}
    response = requests.post(url, json=payload) # request 제출
    return response

# 원하는 input text를 전달한다. 예시는 NLP model이므로 문자열 형태로 전달했다.
rs_rest = get_rest_request(text="classify my text")
rs_rest.json()
```

---

## 8.9 model server inference: gRPC

TODO

---

## 8.10 A/B model test

TODO

---

## 8.11 model server에 model metadata 요청: REST API

TF Serving에서는 model metadata를 제공하는 endpoint를 사용할 수 있게 구성해 준다. 우선 URL은 다음과 같다.

```bash
http://{HOST}:{PORT}/v1/models/{MODEL_NAME}/metadata

# version마다 다른 metadata를 요청하고 싶다면
http://{HOST}:{PORT}/v1/models/{MODEL_NAME}/versions/{MODEL_VERSION}/metadata
```

이 URL을 사용해서 단일 GET 요청으로 model metadata를 얻을 수 있다.

```python
import requests

def metadata_rest_request(model_name, host="localhost",
                          port=8501, version=None):
    url = "http://{}:{}/v1/models/{}/".format(host, post, model_name)
    if version:
        url += "versions/{}".formal(version)
    url += "/metadata"
    response = requests.get(url=url)
    return response
```

해당 request를 받은 server는 model 사양과 정의를 각각 `model_spec`, `metadata` 딕셔너리로 반환한다.

```
{
    "model_spec": {
        "name": "complaints_classification",
        "signature_name": "",
        "version": "1556583584"
    },
    "metadata": {
        "signature_def": {
            signature_def: {
                ...
```

---

## 8.12 model server에 model metadata 요청: gRPC

TODO

---