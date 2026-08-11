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

## Reference
Andrej Karpathy의 [micrograd](https://github.com/karpathy/micrograd)를 
학습 목적으로 재구현하였다.
직접 추가할 부분: PyTorch 그래디언트 대조 검증, 
연산자 확장(exp/log), 의도적 버그 주입 실험.