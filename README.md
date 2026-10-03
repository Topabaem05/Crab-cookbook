<p align="center"><img src="assets/banner.png" alt="ShoreCrab-128M" width="100%"></p>

# Crab-cookbook

ShoreCrab의 이미지 판단을 Python·HTTP로 호출하고 애플리케이션에 연결하는 예제입니다. 이미지, 질문, 2–16개 후보를 입력하면 후보별 확률과 “해당 없음” 확률을 반환합니다.

[모델 카드](https://huggingface.co/Haverbex/ShoreCrab-128M) · [Technical Report](https://app.notion.com/p/3ede9d745acc80b0ab0ec91cd9e96ecb) · [API](docs/api.md) · [활용 사례](docs/use-cases.md)

## 빠른 시작

Python 3.11 이상에서 설치합니다. 클라이언트만 사용하면 모델 실행 라이브러리가 필요하지 않습니다.

```bash
python -m pip install -e .
export CRAB_BASE_URL=http://127.0.0.1:8090
python examples/ask_image.py assets/kitchen.png
```

실행 중인 ShoreCrab 서버 주소가 필요합니다. 외부 서버가 API 키를 요구하면 `CRAB_API_KEY`를 설정하세요.

```python
from shorecrab import CrabClient

client = CrabClient("http://127.0.0.1:8090")
result = client.score(
    "assets/kitchen.png",
    "What color is the mug?",
    ["white", "red", "blue", "black"],
)
print(result["answer"], result["probabilities"], result["none_probability"])
```

## 로컬에서 모델 실행

사용 권한이 있는 A60 추론 번들을 지정합니다. 이 저장소에는 모델 가중치가 포함되지 않으며, 모델 카드 링크는 가중치 다운로드가 제공된다는 의미가 아닙니다.

```bash
python -m pip install -e '.[serve]'
crab-serve --bundle /path/to/a60-inference-bundle --device cpu
```

번들은 `manifest.json`, `model.safetensors`, `tokenizer/`로 구성됩니다. 로더는 매니페스트의 SHA256을 확인하고, 모델을 FP32로 실행합니다. 선택한 장치를 사용할 수 없으면 다른 장치로 자동 전환하지 않습니다.

Apple Silicon에서 GPU를 사용하려면:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 \
crab-serve --bundle /path/to/a60-inference-bundle --device mps
```

## llama.cpp · vLLM에서 직접 실행

vLLM 등록 모델과 llama.cpp의 `libllama` 확장 API를 제공합니다. 이미지·텍스트 인코더와 선택지 점수 계산을 엔진 내부에서 실행합니다. 검증 범위는 Apple Silicon CPU FP32이며, llama.cpp는 전용 확장 빌드가 필요합니다. 설치·변환·호출 명령은 [네이티브 엔진 가이드](docs/native-engines.md)에 있습니다.

## 예제

```bash
python examples/ask_image.py assets/kitchen.png --question '컵은 무슨 색인가요?' --choices 흰색 빨간색 파란색 검은색
python examples/catalogue.py assets/kitchen.png --items 'a mug' 'a chair' 'a phone'
python examples/relative_depth.py assets/kitchen.png 'white mug' 'green apple'
python examples/robot_observation.py assets/kitchen.png
python examples/game_direction.py /path/to/game-frame.png
```

확률은 후보 순서를 따릅니다. `answer`가 `null`이면 “해당 없음”이 가장 높은 것입니다. 현재 값은 보정되지 않은 모델 출력입니다. 범용 A60 호출과 기술 리포트의 게임 전용 헤드 결과를 구분해서 사용하세요.

## License

예제 코드와 추론 코드는 Apache-2.0입니다. 모델·데이터·제삼자 구성요소의 사용 조건은 별도입니다. [NOTICE](NOTICE.md)를 확인하세요.
