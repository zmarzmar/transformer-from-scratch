## 개요
Micrograd란 Autograd(Automatic Differentiation) 즉 자동 미분 엔진으로 Backpropagation(역전파) 알고리즘을 구현한 것
(Karpathy는 영상에서 "automatic gradient"라고 불렀지만 일반적으로 통용되는 표기는 automatic differentiation이다.)

Backpropagation(역전파)는 손실 함수를 가중치로 미분한 기울기를 효율적으로 계산할 수 있도록 해주는 알고리즘이다.

신경망의 가중치를 반복적으로 조정하여 손실 함수를 최소화하여 네트워크의 정확도를 향상시킬 수 있다.


## 알고 가야하는 내용
1. 미분: 한 변수에서의 기울기.
2. 편미분: 여러 변수 중 하나에 대한 기울기.
3. Gradient(기울기): 모든 편미분을 모아 놓은 벡터.
4. Backpropagation(역전파): 연쇄법칙(Chain Rule)을 이용해 출력에서 입력 방향으로 모든 편미분을 효율적으로 계산하는 알고리즘.



## Micrograd에서 grad란?
최종 결과(output)가 이 변수에 얼마나 민감한가

loss = xy에서
x.grad = y
y.grad = x
가 된다.

print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da 즉 a의 증가 기울기는 138
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db 즉 b의 증가 기울기는 645


## 미분
$$L = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

x를 아주 작은 h만큼 증가시켰을 때 함수가 얼마나 반응하는지를 h로 나눠, h와 무관한 값(변화율)으로 만드는 것. h → 0 극한을 취해야 그 값이 하나로 확정된다.


## Value
신경망은 상당히 방대한 수학적 표현식을 다루기 때문에 이러한 표현식을 유지할 데이터 구조가 필요하다.
Value 클래스는 하나의 스칼라 값을 래핑하고 추적한다.

```python
class Value:

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Value(data={self.data})"
```

`__init__`은 객체가 만들어질 때 자동으로 불린다. 넘어온 `data`는 함수가 끝나면
사라지므로, `self.data`에 담아 객체가 살아있는 동안 보존한다.

`__repr__`이 없으면 객체를 출력했을 때 `<__main__.Value object at 0x104f2a3d0>`처럼
메모리 주소만 나온다. 앞으로 Value를 수십 개 만들어 더하고 곱할 텐데 그 상태로는
디버깅이 불가능하다. 그래서 값을 저장하는 것만으로 부족하고 출력 형식까지 정의한다.

1. 객체에 대한 덧셈 연산자
```python
def __add__(self, other):
    out = Value(self.data + other.data)
    return out
```
덧셈 연산자 실행시 파이썬은 내부적으로 a.__add__(b)를 실행한다.

2. 객체에 대한 곱셈 연산자
```python
def __mul__(self, other):
    out = Value(self.data * other.data)
    return out
```
곱셈 연산자 실행시 파이썬은 내부적으로 a.__mul__(b)를 실행한다.

표현식을 연결하는 부분이 필요하다. 즉 어떤 값이 어떤 값을 생성하는지 알고 있어야 한다. 따라서 children이라는 새로운 변수를 사용한다.
```python
class Value:

    def __init__(self, data, _children=()):
        self.data = data
        self._prev = set(_children)

    def __repr__(self):
        return f"Value(data={self.data})"
```
튜플이지만 클래스에서 관리할 때는 집합으로 처리(효율성을 위해)

덧셈이나 곱셈을 통해 값을 만들 때는 해당 값의 자식들을 전달해야 한다.
```python
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other))
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other))
        return out
```

이제 모든 값의 자식은 알지만, 그 값이 어떤 연산으로 만들어졌는지는 알 수 없다.
덧셈으로 나온 4인지 곱셈으로 나온 4인지 구분이 안 된다. 그래서 `_op`를 추가한다.

`_children=()`와 `_op=''`는 둘 다 기본값이 "비어 있음"이다. a, b, c, f처럼 내가 직접
만든 값(리프 노드)은 자식도 없고 생성 연산도 없으므로 아무것도 넘기지 않는다.
반대로 `_op`가 비어 있지 않다는 건 그 노드가 계산으로 생겨났다는 뜻이다.
```python
class Value:
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op=_op
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other),'+')
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
```

이제 완전한 수학적 표현식을 갖게 되었다. 이를 통해 각 값이 어떻게 생성되었는지 다른 값들로부터 어떻게 생성되었는지 정확하게 알 수 있다.

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
```

```python
d._prev
```
결과: {Value(data=-6.0), Value(data=10.0)}

```python
d._op
```
결과: '+'



그래프를 그려보면 노드에 숫자만 나와서 어느 게 a고 어느 게 b인지 구분이 안 된다.
사람이 읽을 이름표가 필요하므로 `label`을 추가한다.
```python
class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op=_op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other),'+')
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

a = Value(2.0,label='a')
b = Value(-3.0,label='b')
c = Value(10.0,label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0,label='f')
L = d * f; L.label='L'
```

이 표현식을 그림으로 그려주는 `draw_dot` 함수. `trace`가 그래프를 훑으며 노드와 간선을
모으고, `draw_dot`이 그걸 graphviz로 그린다.

```python
from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right

  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f }" % (n.label,n.data), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

draw_dot(L)
```

![draw_dot(L) 실행 결과 — a*b+c 계산 그래프](../../results/01-graph.svg)

이 모든 과정은 순방향 전달의 출력을 나타낸 것 

역전파를 실행시 모든 중간 값들을 따라 기울기를 계산
이때 각 값에 대해 계산하는 건 노드 L에 대한 미분

신경망 설정에서는 손실 함수 L을 신경망의 가중치에 대해 미분하는 것이 매우 중요하다
a,b,c,d,f 는 모두 가중치. 따라서 이러한 가중치가 손실 함수에 어떤 영향을 미치는지 알아야 한다.

이 신경망의 리프노드들은 신경망의 가중치가 되고 다른 리프노드들은 데이터가 된다.
## 직접 해볼 것
- [ ] PyTorch 그래디언트와 대조 검증
- [ ] 연산자 확장 (exp, log)
- [ ] 의도적 버그 주입 실험 (grad를 `=`로 덮어쓰면 뭐가 깨지나)