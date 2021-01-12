# 基于路段通行时间预测的多模态交通出行推荐
A multi-mode route recommendation implementation based on travel time estimation on Python.

## 环境配置(requirements)

### 必选
```yaml
python>=3.7
numpy
pandas
geopandas
networkx
PyMetis
tensorboardX
tqdm
requests
scipy
Shapely
statsmodels
pyproj
dgl
pytorch
fastparquet
```

### 可选
```yaml
flask
matplotlib
seaborn
```

## 模型训练及测试

模型训练和测试在北京数据集上进行，数据集的加载和处理代码参考`utils.bj_dataset.BJSpeedDataset`，模型的代码和模型配置超参数参考`models.ours`。模型训练和测试的参数如下：

```text
usage: main.py [-h] --save_folder SAVE_FOLDER --cuda CUDA [--loss LOSS] [--optimizer OPTIMIZER] [--lr LR] [--max_grad_norm MAX_GRAD_NORM] [--epochs EPOCHS]
               [--early_stop_steps EARLY_STOP_STEPS] [--test] [--resume]

optional arguments:
  -h, --help            show this help message and exit
  --save_folder SAVE_FOLDER
                        where to save the logs and results.
  --cuda CUDA           the torch device indicator, like "cuda:0".
  --loss LOSS           loss function, default as MaskedMAELoss.
  --optimizer OPTIMIZER
                        optimizer name, default as `Adam`
  --lr LR               the learning rate.
  --max_grad_norm MAX_GRAD_NORM
                        clips gradient norm of an iterable of parameters.
  --epochs EPOCHS       the max number of epochs to train.
  --early_stop_steps EARLY_STOP_STEPS
                        steps to early stop.
  --test                toggle the test mode.
  --resume              if to resume a trained model.

```

### 模型训练

```sh
python main.py --save_folder myFolder --cuda cuda:0
```

### 模型测试

```sh
python main.py --save_folder myFolder --cuda cuda:0 --test
```

## 路线规划

### 基于A*算法的依赖时间的最短路径搜索

```python
def time_dependent_astar_path(graph, source: TrafficNode, target: TrafficNode,
                              heuristic=None, weight="weight", stime=0.0) -> List[TrafficNode]:
    """Returns a list of nodes in a shortest path between source and target
    using the time dependent A* ("A-star") algorithm.

    There may be more than one shortest path.  This returns only one.

    Parameters
    ----------
    graph : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    heuristic : a function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.

    weight : a function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.
       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly four
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge, and a cost from
       the source node to the start point of the edge. The function
       must return a number.

    stime : a float
        If it is provided, that is the pre-accumulated cost of the route.
    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.
    """
```

通过指定`weight`参数可以指定不同的权重参数。基于此参数，可以按照预先计算好的未来路网速度的预测值或实时计算路网参数，来得到动态的路网权重。在最短路径搜索的过程中，边的权值就是`weight`函数返回的值。该函数必须精确地接受四个位置参数：边的起点、边的终点、边的边属性字典，以及从源节点到该边的起点的累计代价。`weight`函数必须返回一个浮点数。

### 代码的使用和返回值说明

使用demo：

```python
import datetime
import random

from shapely.geometry import Point

from utils.routes import RoutePlanner


def random_point_within(poly):
    min_x, min_y, max_x, max_y = poly.bounds
    while True:
        x, y = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
        if poly.contains(Point([x, y])):
            return x, y

def test(region, method):


    src, dst = random_point_within(region), random_point_within(region)
    t = datetime.datetime.utcnow().timestamp() + random.randint(-24 * 60 * 60, +24 * 60 * 60)

    return method(src, dst, t)

rp = RoutePlanner()

results = test(rp.shenzhen, rp)
```

返回值为：

```python
RoutingResult = namedtuple(
    'RoutingResult', [
        'info',  # 成功状态说明
        'origin',  # 起点坐标
        'destination',  # 终点坐标
        'timestamp',  # 出发时间
        'plans'  # 换乘方案列表
    ]
)
```

其中`'info'`为字符串类型，说明了返回值的状态，目前由如下三种成功状态：

```python
INFO_SUCCESS = 'success'
INFO_UNREACH = 'error: not reachable.'
INFO_TOO_CLOSE = 'error: origin and destination are too close.'
```

`'origin'`和`'destination'`是表示起点和终点的坐标的二元组；`'timestamp'`是表示出发时间的UNIX时间戳，即从1970年1月1日起到对应时间的秒数；`'plans'`是一个`RoutingPlan`的对象或者`RoutingPlan`的列表，在需要在不同交通方式之间换乘时，不同的交通方式所对应的子路径分别对应一个`RoutingPlan`列表。`RoutingPlan`的说明如下：

```python
RoutingPlan = namedtuple(
    'RoutingPlan', [
        'cost',  # 此换乘方案价格，单位：元
        'time',  # 预期时间，单位：秒
        'distance',  # 此换乘方案全程距离，单位：米
        'walking_distance',  # 此方案总步行距离，单位：米
        'transit_distance',  # 此方案公交行驶距离，单位：米
        'taxi_distance',  # 此方案出租车行驶距离，单位：米
        'path',  # 此换乘方案的路径坐标列表
        'segments'  # 换乘路段列表，以每次换乘动结束作为分段点，将整个换乘方案分隔成若干 Segment（换乘路段）
    ]
)
```

### 模式交通网络的构建

```python
from utils.routes import subway, bus, road_net


subway(shentie)  # 从"https://www.szmc.net/styles/index/sz-subway/mcdata/shentie.json"构建地铁网络
bus(route_path, stop_path, route_stop_path) # 从公交路线、站点、对应关系构建公交网络
road_net(path, predict_speed)  # 从道路shapefile文件和路网速度预测方法构建出租车交通网络
```

其中`predict_speed`函数精确地接受两个参数：1) 车辆在该条道路上行驶的开始时间戳，2) `road_id`道路的唯一标识代码；并返回一个数字表示预测的对应路段在对应时间的路段车流速度，单位为`km/h`。