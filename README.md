# feature_hack

### Данное решение используется для:
1) определения оптимального маршрута на сетевой структуре с помощью алгоритма Дейкстры
2) вычисления расстояния хаверсайна для взвешивания ребёр сетевой структуры
3) нейросетевой суммаризации подробного описания объекта (`text-to-text трансформер T5`)
4) парсиннг подробного описания объекта с помощью wiki api
5) определения точек взаимодействия с логикой работы программного кода

### Технологический стек
- язык: `python`
- библиотеки: `nest_asyncio`, `numpy`, `pandas`, `requests`, `uvicorn`, `fastapi`, `pydantic`, `pyngrok`, `sklearn`, `transformers`, `wikipedia`
