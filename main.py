
# ПРАКТИЧЕСКОЕ ЗАНЯТИЕ № 1
# Разработка REST API для AI-сервиса на FastAPI
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\venv\Scripts\Activate.ps1
# python main.py

# ------------------------------------------------------------
# ИМПОРТ БИБЛИОТЕК
# ------------------------------------------------------------

import time
import uuid
import logging
from typing import Optional, List
import fastapi




from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    status
)

from pydantic import (
    BaseModel,
    Field
)


# ------------------------------------------------------------
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# FASTAPI-ПРИЛОЖЕНИЕ
# ------------------------------------------------------------

app = FastAPI(
    title="Agro Scoring API",
    description=(
        "REST API для оценки риска "
        "сельскохозяйственных предприятий."
    ),
    version="1.0.0"
)


# ------------------------------------------------------------
# КОНСТАНТЫ
# ------------------------------------------------------------

MODEL_NAME = "agro-risk-model"
MODEL_VERSION = "1.0"
MODEL_TYPE = "risk-scoring"

# Имитируем состояние модели.
# Если установить False, /predict будет возвращать HTTP 503.
MODEL_READY = True


# Разрешенные регионы.
# Это пример бизнес-справочника.
ALLOWED_REGIONS = {
    "Krasnodar",
    "Rostov",
    "Stavropol"
}


# ------------------------------------------------------------
# ВРЕМЕННОЕ ХРАНИЛИЩЕ
# ------------------------------------------------------------

# Позже подключим PostgreSQL или SQLite.
# Пока будем использовать словарь predictions 

predictions = {}

# ------------------------------------------------------------
# PYDANTIC-МОДЕЛЬ ВХОДНЫХ ДАННЫХ
# ------------------------------------------------------------

class FarmRequest(BaseModel):
    """
    Данные сельскохозяйственного предприятия,
    которые клиент передает в POST /predict
    """

    farm_id: str = Field(
        ...,
        min_length=1,
        description="Идентификатор хозяйства"
    )

    region: str = Field(
        ...,
        min_length=1,
        description="Регион хозяйства"
    )

    crop_type: str = Field(
        ...,
        min_length=1,
        description="Основная сельскохозяйственная культура"
    )

    area_ha: float = Field(
        ...,
        gt=0,
        description="Площадь посевов, га"
    )

    temperature_avg: float = Field(
        ...,
        ge=-60,
        le=60,
        description="Средняя температура, °C"
    )

    precipitation_mm: float = Field(
        ...,
        ge=0,
        description="Количество осадков, мм"
    )

    payment_delay_days: int = Field(
        ...,
        ge=0,
        description="Количество дней просрочки платежа"
    )

    previous_defaults: int = Field(
        ...,
        ge=0,
        description="Количество предыдущих дефолтов"
    )

    debt: float = Field(
        ...,
        ge=0,
        description="Текущая задолженность"
    )

# ------------------------------------------------------------
# PYDANTIC-МОДЕЛЬ ОТВЕТА
# ------------------------------------------------------------

class PredictionResponse(BaseModel):
    """
    Структура ответа сервиса после выполнения прогноза.
    """

    request_id: str
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str


# ------------------------------------------------------------
# МОДЕЛЬ ОТВЕТА /health
# ------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str


# ------------------------------------------------------------
# МОДЕЛЬ ОТВЕТА /model-info
# ------------------------------------------------------------

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: str


# ------------------------------------------------------------
# МОДЕЛЬ МАШИННОГО ОБУЧЕНИЯ

# Позже подключим обученную ML модель
# Сейчас определим функцию с условной конструкцией
# ------------------------------------------------------------

def calculate_risk(data: FarmRequest) -> float:
    """
    Реализует простую логику оценки риска по финансовым показателям
    и отраслевым факторам.
    """
        
    score = 0.1

    # Длительная просрочка увеличивает риск
    if data.payment_delay_days > 30:
        score += 0.3

    # Наличие предыдущих дефолтов увеличивает риск
    if data.previous_defaults > 0:
        score += 0.3

    # Большая задолженность увеличивает риск
    if data.debt > 5_000_000:
        score += 0.2

    # Малое количество осадков условно увеличивает
    # аграрный риск
    if data.precipitation_mm < 100:
        score += 0.1

    # Ограничиваем значение диапазоном от 0 до 1
    score = min(score, 1.0)

    return round(score, 2)


# ------------------------------------------------------------
# ПОСТПРОЦЕССИНГ:
# ПОЛУЧЕНИЕ РИСК-ПРОФИЛЯ
# ------------------------------------------------------------

def get_risk_level(score: float) -> str:
    """
    Преобразует score в категорию риска.
    """

    if score < 0.3:
        return "low"

    if score < 0.7:
        return "medium"

    return "high"


# ------------------------------------------------------------
# ПОСТПРОЦЕССИНГ:
# ВЫДАЧА РЕКОМЕНДАЦИИ
# ------------------------------------------------------------

def get_recommendation(level: str) -> str:
    """
    Возвращает рекомендацию в зависимости от категории риска.
    """

    if level == "low":
        return "Стандартное рассмотрение"

    if level == "medium":
        return "Требуется дополнительная проверка"

    return "Высокий риск. Требуется ручное рассмотрение"


# ------------------------------------------------------------
# MIDDLEWARE
# Измеряем время обработки каждого HTTP-запроса.
# ------------------------------------------------------------

@app.middleware("http")
async def add_process_time(
    request: Request,
    call_next
):
    """
    Middleware выполняется для каждого HTTP-запроса.
    Добавляет в HTTP-ответ заголовок:

        X-Process-Time

    содержащий время обработки запроса.
    """

    start_time = time.perf_counter()

    response = await call_next(request)

    process_time = time.perf_counter() - start_time

    response.headers["X-Process-Time"] = str(
        round(process_time, 6)
    )

    return response


# ------------------------------------------------------------
# ENDPOINTS:
# ------------------------------------------------------------


# ------------------------------------------------------------
# GET /health
# ------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния API",
    description=(
        "Используется для проверки того, "
        "что REST API запущен и отвечает."
    )
)
def health():
    """
    Проверка работоспособности API.
    """

    return {
        "status": "ok"
    }


# ------------------------------------------------------------
# GET /model-info
# ------------------------------------------------------------

@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Информация о модели",
    description=(
        "Возвращает название, версию, тип "
        "и текущее состояние модели."
    )
)
def model_info():
    """
    Возвращает информацию
    об используемой ML-модели.
    """

    model_status = (
        "ready"
        if MODEL_READY
        else "unavailable"
    )

    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "status": model_status
    }


# ------------------------------------------------------------
# POST /predict
# ------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Оценить риск хозяйства",
    description=(
        "Принимает характеристики хозяйства, "
        "выполняет валидацию, инференс модели "
        "и возвращает оценку риска."
    )
)
def predict(request: FarmRequest):
    """
    Основной endpoint сервиса.
    """

    # 1. Проверяем состояние ML-модели
   
    if not MODEL_READY:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is temporarily unavailable"
        )


    # 2. Бизнес-валидация
    #
    # Pydantic уже проверил:
    # - типы данных;
    # - обязательность;
    # - диапазоны.
    #
    # Осталось проверить бизнес-правила (например, разрешённые регионы).
  
    if request.region not in ALLOWED_REGIONS:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown region: {request.region}. "
                f"Allowed regions: "
                f"{sorted(ALLOWED_REGIONS)}"
            )
        )
        
    # 3. Логирование входящего запроса
   
    logger.info(
        "Prediction request received | farm_id=%s",
        request.farm_id
    )

  
    # 4. ИНФЕРЕНС
    
    score = calculate_risk(request)

    # 5. ПОСТПРОЦЕССИНГ
 
    level = get_risk_level(score)

    recommendation = get_recommendation(level)

    # 6. СОЗДАЕМ УНИКАЛЬНЫЙ ID ЗАПРОСА
    
    request_id = str(uuid.uuid4())

    # 7. ФОРМИРУЕМ РЕЗУЛЬТАТ
 
    result = {
        "request_id": request_id,
        "farm_id": request.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": recommendation,
        "model_version": MODEL_VERSION
    }


    # 8. СОХРАНЯЕМ РЕЗУЛЬТАТ
        
    predictions[request_id] = result

    # 9. ЛОГИРОВАНИЕ РЕЗУЛЬТАТА
 
    logger.info(
        "Prediction completed | "
        "request_id=%s | "
        "farm_id=%s | "
        "risk_score=%s | "
        "risk_level=%s",
        request_id,
        request.farm_id,
        score,
        level
    )

    # 10. ВОЗВРАЩАЕМ HTTP-ОТВЕТ

    return result


# ------------------------------------------------------------
# GET /predictions
#
# этот endpoint должен быть расположен ДО
# /predictions/{request_id}
# ------------------------------------------------------------

@app.get(
    "/predictions",
    response_model=List[PredictionResponse],
    summary="Получить список прогнозов",
    description=(
        "Возвращает список выполненных прогнозов. "
        "Поддерживает ограничение количества результатов "
        "и фильтрацию по уровню риска."
    )
)
def get_predictions(

    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Максимальное количество результатов"
    ),

    risk_level: Optional[str] = Query(
        default=None,
        description=(
            "Фильтр по категории риска: "
            "low, medium или high"
        )
    )

):
    """
    Получение списка прогнозов.
    """

    # Преобразуем словарь в список
    values = list(predictions.values())

    # Проверяем корректность risk_level

    allowed_levels = {
        "low",
        "medium",
        "high"
    }

    if risk_level is not None:

        if risk_level not in allowed_levels:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "risk_level must be "
                    "'low', 'medium' or 'high'"
                )
            )

        # Фильтрация
      
        values = [
            item
            for item in values
            if item["risk_level"] == risk_level
        ]

    # Ограничение количества результатов
  
    return values[:limit]


# ------------------------------------------------------------
# GET /predictions/{request_id}
# ------------------------------------------------------------

@app.get(
    "/predictions/{request_id}",
    response_model=PredictionResponse,
    summary="Получить прогноз по request_id",
    description=(
        "Возвращает сохраненный прогноз "
        "по его уникальному идентификатору."
    )
)
def get_prediction(request_id: str):
    """
    Получение одного прогноза
    по request_id.
    """

    # Проверяем наличие прогноза

    if request_id not in predictions:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )

    # Возвращаем прогноз
 
    return predictions[request_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)