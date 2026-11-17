import os

import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

st.set_page_config(page_title="AI 번역기", page_icon="🌐", layout="centered")

LANGUAGES = {"영어": "English", "일본어": "Japanese", "스페인어": "Spanish"}
# 화면 표시용. 국기 이모지는 Windows에서 글자(US/JP)로 깨져 보여서 언어 코드를 쓴다.
FLAGS = {"영어": "EN", "일본어": "JA", "스페인어": "ES"}
MAX_CHARS = 5000
DEFAULT_MODEL = "gpt-6.1-sol"
# 기본값(읽기 600초, 재시도 2회)이면 API가 멈출 때 화면이 오래 대기하므로 제한한다.
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 1

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text into {language}. "
    "Output only the translated text, with no explanations, quotes, or notes. "
    "Preserve line breaks, tone, and formatting."
)


class TranslationError(Exception):
    """사용자에게 그대로 보여줄 수 있는 메시지를 가진 번역 오류."""


def get_setting(name: str, default: str | None = None) -> str | None:
    """환경변수(.env 포함) → st.secrets 순으로 설정값을 조회한다."""
    value = os.getenv(name)
    if value:
        return value
    try:
        value = st.secrets.get(name)
    except Exception:  # secrets.toml이 없으면 예외가 발생할 수 있다
        value = None
    return value or default


def translate(text: str, target_language: str) -> str:
    api_key = get_setting("OPENAI_API_KEY")
    if not api_key:
        raise TranslationError(
            "API 키가 설정되지 않았습니다. .env 파일(또는 배포 환경의 Secrets)에 "
            "OPENAI_API_KEY를 등록해 주세요."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=get_setting("OPENAI_BASE_URL"),
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )
    model = get_setting("OPENAI_MODEL", DEFAULT_MODEL)

    try:
        response = client.chat.completions.create(
            model=model,
            # gpt-6.1-sol은 temperature 기본값(1)만 지원하므로 지정하지 않는다.
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(language=target_language),
                },
                {"role": "user", "content": text},
            ],
        )
    except AuthenticationError:
        raise TranslationError("API 키 인증에 실패했습니다. 키가 올바른지 확인해 주세요.")
    except RateLimitError:
        raise TranslationError("요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.")
    except NotFoundError:
        raise TranslationError(
            f"모델 '{model}'을(를) 찾을 수 없습니다. OPENAI_MODEL 설정을 확인해 주세요."
        )
    except APITimeoutError:
        raise TranslationError(
            f"응답이 {REQUEST_TIMEOUT_SECONDS}초 안에 오지 않았습니다. 잠시 후 다시 시도해 주세요."
        )
    except APIConnectionError:
        raise TranslationError("서버에 연결할 수 없습니다. 네트워크 상태를 확인해 주세요.")
    except BadRequestError as e:
        raise TranslationError(f"요청이 올바르지 않습니다. ({e.message})")
    except InternalServerError:
        raise TranslationError("번역 서버에 일시적인 문제가 있습니다. 잠시 후 다시 시도해 주세요.")
    except APIStatusError as e:
        raise TranslationError(f"번역 요청이 실패했습니다. (HTTP {e.status_code})")

    if not response.choices:
        raise TranslationError("번역 결과를 받지 못했습니다. 다시 시도해 주세요.")
    content = response.choices[0].message.content
    if not content:
        raise TranslationError("번역 결과가 비어 있습니다. 다시 시도해 주세요.")
    return content.strip()


st.markdown(
    """
    <style>
    .block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 720px; }
    h1 { text-align: center; }
    div.stButton > button { height: 3rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌐 AI 번역기")
st.markdown(
    "<p style='text-align:center; opacity:0.65'>텍스트를 입력하고 번역할 언어를 선택하세요.</p>",
    unsafe_allow_html=True,
)

text = st.text_area(
    "번역할 텍스트",
    height=220,
    key="input_text",
    placeholder="여기에 번역할 글을 입력하세요... (입력 후 입력창 밖을 클릭하거나 Ctrl+Enter를 누르면 글자 수가 갱신됩니다)",
    label_visibility="collapsed",
)

counter = f"{len(text):,} / {MAX_CHARS:,}"
st.markdown(
    f"<div style='text-align:right; font-size:0.85rem; opacity:0.7'>"
    f"{counter if len(text) <= MAX_CHARS else '⚠️ ' + counter}</div>",
    unsafe_allow_html=True,
)

clicked_language = None
cols = st.columns(len(LANGUAGES))
for col, (label, target) in zip(cols, LANGUAGES.items()):
    if col.button(
        f"{FLAGS[label]} {label}", type="primary", use_container_width=True
    ):
        clicked_language = (label, target)

if clicked_language:
    label, target = clicked_language
    st.session_state.pop("result", None)
    if not text.strip():
        st.warning("번역할 텍스트를 입력하세요.")
    elif len(text) > MAX_CHARS:
        st.warning(
            f"입력은 최대 {MAX_CHARS:,}자까지 가능합니다. (현재 {len(text):,}자)"
        )
    else:
        try:
            with st.spinner(f"{label}로 번역하는 중..."):
                translated = translate(text, target)
            st.session_state["result"] = (label, translated)
        except TranslationError as e:
            st.error(str(e))

if "result" in st.session_state:
    label, translated = st.session_state["result"]
    st.divider()
    st.subheader("번역 결과")
    with st.container(border=True):
        st.caption(f"{FLAGS[label]} {label}로 번역됨")
        st.code(translated, language=None, wrap_lines=True)
