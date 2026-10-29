
title: On Policy Distillation - Recent trend
speaker: Marco 


## Overview
* motivation:
	* 우리팀은 그간 SFT을 통해서 LLM /CE를 학습 해왔다. 
	* 두가지 관점에서 학습 방법론을 업그레이드 할 필요성을 생각한다.
		* 첫번째 리스닝 모델링을 도입해야한다. 하지만 현재 sampling by teacher and SFT distillation방식은 한계가 보인다.
		* 둘째 distillation 학습 효율성 개선이다. data scaling을 해도 모델 성능 개선에 한계가 보이기 시작했다
		* 셋째 작은 sLLM으로는 distillation이 잘 안된다. 하지만 online modeling을 위해서는 1B under의 모델을 학습 시켜야 한다.
		* 셋째 기존 지식을 유지하면서도 새로운 지식의 주입하는 새로운 방법론이 필요하다

relevnace LL/CE SFT
- 시퀀스 출처 X: 데이터 샘플링
- 채점자 Y: teacher LLM (soft) or human (hard)
- 채점시점: episode 종류 후 매번 (reasoning X)
- episode당 신호: O(1) bits
- 학습중 teacher: 불필요
- 샘플링 비용: 없음 (reasoning X)
- 시퀀스 길이 : 데이터셋 그대로
- batch size: 크게

우리는 결국 teacher가 리즈닝을 해야한다.
- 그리고 그 스코어를 student CE가 학습 해야한다.
teacher LLM을 더 큰 LLM으로 부터 학습 시켜야 한다.
