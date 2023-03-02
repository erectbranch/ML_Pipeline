## 1 Microsoft MLOps REST API 사용법

> [AML COMPUTE REST API SAMPLE](https://github.com/microsoft/MLOps/blob/master/examples/AzureML-REST-API/compute.md)

REST API 코드 작성 전 먼저 준비해야 하는 절차는 다음과 같다.

- Azure Subscription 허가를 가지고 있어야 한다.

- Azure ML Workspace를 생성해 두어야 한다.

- (local) Python library `requests`와 `adal`(Azure AD 인증 library)을 설치해야 한다.

---

### 1.1 Client ID, Secret 얻기

> [Azure Portal](https://github.com/microsoft/MLOps/blob/master/examples/AzureML-REST-API/portal.azure.com): Activate Directory > App Registration > New Registration

위 Azure Portal 링크에 접속해서 client ID와 secret을 얻어야 한다. 혹은 터미널 명령(Azure CLI)으로도 얻을 수 있다.

```bash
az ad sp create-for-rbac --sdk-auth --name <my-sp-name>

# 출력은 다음과 같이 나온다.
# {
# 	"clientId": "<my-client-id>",
# 	"clientSecret": "<my-clent-secret>",
# 	....
#}
```

---

### 1.2 Azure Authenticate

이제 권한을 인증해야 한다. 

- `client_id`, `client_secret`: 앞서 확인한 client ID와 secret

- `user_name`: login 때 사용하는 user name

- `subid`: Azure subscription ID

- `rg`: workspace 정보

위 정보를 바탕으로 `adal` library를 통해 인증 request를 보내고, response로 받은 token을 사용한다.

```python
import requests
import json
import time
from adal import AuthenticationContext

client_id = "<my-client-id>"
client_secret = "<my-clent-secret>"
user_name = "<my-user-name>"

subid = "<my-subscription-id>"
rg = "<my-workspace-resource-group>"

auth_context = AuthenticationContext("https://login.microsoftonline.com/{}.onmicrosoft.com".format(user_name))

resp = auth_context.acquire_token_with_client_credentials("https://management.azure.com/",client_id,client_secret)

token = resp["accessToken"]
```

---

### 1.3 Create/Update AML Compute

계산을 진행하기 위해서 필요한 준비물은 다음과 같다.

- 1.2절에서 response로 받은 token

- 미리 생성한 AML workspace

- 미리 생성한 Azure Virtual Network

준비물이 다 갖춰졌다면 코드를 작성해 보자.

1. 일반 config 설정

```Python
# general config
subid = "<my-subscription-id>"
rg = "<my-workspace-resource-group>"
ws = "<my-workspace-name>"
api_version = "2019-06-01"
```

2. compute resource config 설정

```Python
# location 
# 예시: westus2, eastus
location = "<your workspace location>" 

 # compute target 설정
 # 예시: VirtualMachine, AmlCompute
compute_type = "<compute type>"

 # Azure에서 제공하는 VM 종류 설정
 # 예시: Standard_D1, Standard_F64s_v2
vmSize = "<VmType>"

# 해당 ssh 계정과 비밀번호
admin_user_name= "<admin>" 
admin_user_passwd= "<nimda123@S>" 

# Priority 설정
# Dedicated 혹은 Low Priority
vmPriority = "<Dedicated>" 

# basic node의 schedule policy 설정
maxNodeCount = <maxNumber>
minNodeCount = <lowNumner>

# node의 idle time tolerance 설정
# time format은 ISO8601. 아래 세 가지 에시 참고.
# 1 hour 30 minutes --> PT1H30M
# 5 minutes --> PT5M
# 2 hours --> PT2H
nodeIdleTimeBeforeScaleDown = "PT5M" #  set 5 minutes tolerance

# 미리 만든 virtual network id
# 잘 모르겠으면 링크 설명 참고: https://docs.microsoft.com/en-us/azure/virtual-network/virtual-networks-overview
network_id = "<virtual network id>"

# subnet id 혹은 virtual network
subnet_id =  "<default>"
```

3. request 보내기

```Python
body = {
          "location": location, 
          "properties": {
            "computeType": compute_type,
            "properties": {
              "vmSize": vmSize,
              "vmPriority": vmPriority,
              "scaleSettings": {
                "maxNodeCount": maxNodeCount,
                "minNodeCount": minNodeCount,
                "nodeIdleTimeBeforeScaleDown": nodeIdleTimeBeforeScaleDown
              }
            }
          },
          "userAccountCredentials":{
            "adminUserName": admin_user_name,
            "adminUserPassword": admin_user_passwd
          },
          "subnet":{
            "id":"/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Network/virtualNetworks/{}/subnets/{}".format(subid, rg, network_id, subnet_id)
         }
        }

header = {'Authorization': 'Bearer ' + token, "Content-type": "application/json"}
create_compute_url = "https://management.azure.com/subscriptions/{}/resourceGroups/{}/providers/Microsoft.MachineLearningServices/workspaces/{}/computes/{}?api-version={}".format(subid, rg, ws, compute_name, api_verison)

resp = requests.put(create_compute_url, headers=header, json=body)
print resp.text
```

---

### 1.4 Delete AML Compute

준비물은 Create/Update AML Compute 문단과 동일하다. request 코드는 다음과 같다.

```Python
subid =  "<my-subscription-id>" 
rg =  "<my-workspace-resource-group>" 
ws =  "<my-workspace-name>" 
api_version =  "2019-06-01"

# 삭제할 compute name 
compute_name = "<to be delete compute name>"

# Action 지정(Delete 혹은 Detach)
#'Delete': workspace에서 underlying compute를 삭제한다.
#'Detach': workspace에서 underlying compute를 분리한다.
underlying_resource_action = "<Delete>"

header = {'Authorization': 'Bearer ' + token, "Content-type": "application/json"}
delete_compute_url = "https://management.azure.com/subscriptions/{}/resourceGroups/{}/providers/Microsoft.MachineLearningServices/workspaces/{}/computes/{}?api-version={}&underlyingResourceAction={}".format(subid, rg, ws, compute_name, api_verison, underlying_resource_action)

resp = requests.delete(delete_compute_url, headers=header)
print resp.text
```

---

### 1.5 Get AML Compute

준비물은 Create/Update AML Compute 문단과 동일하다. request 코드는 다음과 같다.

```Python
subid =  "<my-subscription-id>" 
rg =  "<my-workspace-resource-group>" 
ws =  "<my-workspace-name>" 
api_version =  "2019-06-01"

# the compute name 
compute_name = "<compute name>"

header = {'Authorization': 'Bearer ' + token, "Content-type": "application/json"}
get_compute_url = "https://management.azure.com/subscriptions/{}/resourceGroups/{}/providers/Microsoft.MachineLearningServices/workspaces/{}/computes/{}?api-version={}".format(subid, rg, ws, compute_name, api_verison)

resp = requests.get(get_compute_url, headers=header)
print resp.text
```

---

### 1.6 List AML Compute By Workspace

준비물은 token과 미리 만든 AML workspace이다. request 코드는 다음과 같다.

```Python
subid =  "<my-subscription-id>" 
rg =  "<my-workspace-resource-group>" 
ws =  "<my-workspace-name>" 
api_version =  "2019-06-01"

header = {'Authorization': 'Bearer ' + token, "Content-type": "application/json"}
list_compute_by_workspace_url = "https://management.azure.com/subscriptions/{}/resourceGroups/{}/providers/Microsoft.MachineLearningServices/workspaces/{}/computes?api-version={}".format(subid, rg, ws, api_verison)

resp = requests.get(list_compute_by_workspace_url , headers=header)
print resp.text
```

---